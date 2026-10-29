# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather one prefill chunk's DSA indexer keys back out of the indexer K cache."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: Mistral Large 4's indexer cache: 128 fp8 bytes plus one fp32 scale per token, 32
#: tokens per page. A page stores all 32 tokens' data first, then their scales. In its cells
#: `layout` says how the view is made: "v2" and "v1" rebuild the per-layer
#: views the two DSA cache managers hand out (see `note`), "manager_v2" and
#: "manager_v1" take the view and the block ids from a real manager.
#: `pool_layers` is the number of indexer rows in the pool, `pool_row` the row
#: gathered from, `num_blocks` the pool's slot count; `tail` puts the step's
#: pages in the pool's last blocks, and `cycle` lets the step's pages outnumber
#: the pool (they are drawn as consecutive fresh permutations of it, so pages
#: repeat but no stretch of the mapping repeats another). `decode_rows` appends
#: that many decode sequences' new-token slots to the mapping after the
#: prompts', as a mixed step does, at drawn positions; `decode_pos` at the
#: positions it lists. `sweep=True` reads every key count of the window as its
#: prefix and its suffix; `sweep="windows"` reads, besides the window, each
#: q-split rank's share of every request of 32768 keys or more, and windows of
#: 2^k - 1, 2^k and 2^k + 1 keys (k = 15..21) at the mapping's start and end and
#: from keys 2^20 and 2^21 (each +-1); `qsplit_ranks` is the number of ranks the
#: split cuts a chunk over (4 unless given). `max_seq_len` sizes a manager
#: (default 9216); `large` marks a cell pre-merge deselects. `cached` gives each
#: prompt a cached prefix ahead of its `seq_lens` new keys: the mapping then holds
#: the prompt's cached and new keys (positions 0..cached + new - 1), as the full-KV
#: mapping of a chunked-prefill step does, and a decode row owns only the page
#: its new key sits on. `sweep_counts` reads each listed key count of the window
#: as its prefix and its suffix.
_ML4 = dict(head_dim=128, tokens_per_block=32)

#: 16 prompts, the most one step schedules (max_batch_size 16), filling the FP8
#: target's 9216-token budget; the first is the shortest prompt the indexer
#: runs on (one key past index_topk 2048).
_ML4_16_PROMPTS = (2049, 1500, 1024, 977, 700, 600, 512, 400, 333, 300, 250, 200, 150, 100, 64, 57)

#: The largest pools a rank can hold, deliberate overestimates: about 24.7 GiB
#: is left per GB300 after the 252.84 GiB of weights, and the managers measure
#: 47280 bytes per token with 44 indexer rows and 50976 with 72, so at most
#: about 17.5K and 16.3K blocks of 32 tokens.
_ML4_FAR_44 = dict(_ML4, pool_layers=44, num_blocks=18432, tail=True)
_ML4_FAR_72 = dict(_ML4, pool_layers=72, num_blocks=16384, tail=True)

#: The NVFP4 target serves max_seq_len = max_num_tokens = 32768 with chunked
#: prefill off: 16 prompts packed to 32768 keys, the first again one key past
#: index_topk, the others across page edges (4097, 4096, 1023, 255, 31).
_ML4_32K_16 = (
    2049,
    9000,
    4097,
    4096,
    3000,
    2500,
    2048,
    2047,
    1024,
    1023,
    777,
    513,
    255,
    64,
    31,
    244,
)
#: Its manager cells: a quota of two full prompts, at its sequence and token limits.
_ML4_32K_MANAGER = dict(_ML4, pool_layers=44, max_tokens=65536, max_seq_len=32768)
#: The largest pool an NVFP4 tep4 rank can hold, a deliberate overestimate: the
#: GB300's 276.62 GiB less 143.24 GiB of weights is 133.38 GiB, 94661 blocks
#: at the manager's 47280 bytes per token (44 indexer rows), and V2 hands out
#: about 5.3% more slots than its quota; 109568 slots clears both. Its 44-row
#: indexer pool is 20.4 GB, so these cells are `large`.
_ML4_32K_FAR = dict(_ML4, pool_layers=44, num_blocks=109568, tail=True, large=True)
#: The largest pool an NVFP4 tep8, tp8 or dep8 rank can hold, a deliberate
#: overestimate: a tep8 or tp8 rank holds 73.61 GiB of weights and a dep8 rank
#: 89.29 GiB (its 16 routed experts and the replicated rest; computed from the
#: shard headers), which leave 203.01 and 187.33 GiB, 144075 and 132949 slots at
#: the manager's 1512960 bytes per 32-token slot. V2 backs every slot it hands
#: out, the ~5% past its quota included, with device memory, so 147456 slots
#: clears both readings. Its 44-row indexer pool is 27.4 GB, so this cell is
#: `large`.
_ML4_32K_8GPU_FAR = dict(_ML4, pool_layers=44, num_blocks=147456, tail=True, large=True)

#: The NVFP4 target's cells at 64-token pages: 8448-byte pages, data at `(pos % 64) * 128`,
#: scales from byte 8192. The same prompts and windows; each pool holds its 32-token
#: counterpart's tokens in half the blocks (1024, 4096, and 54784 for the far pool, the
#: same 20.4 GB).
_ML4_64 = dict(head_dim=128, tokens_per_block=64)
_ML4_32K_P64_MANAGER = dict(_ML4_64, pool_layers=44, max_tokens=65536, max_seq_len=32768)
_ML4_32K_P64_FAR = dict(_ML4_64, pool_layers=44, num_blocks=54784, tail=True, large=True)

#: The NVFP4 tep4 target at its own context ("1M" cells): max_seq_len 1048576 with
#: chunked prefill in 32768-token steps, on V2 views of the 44-row indexer pool.
_ML4_1M_V2 = dict(_ML4, layout="v2", pool_layers=44)
#: The most keys one step's full-KV mapping holds: every token of the delivered
#: tep4 boot's pool (2373024 tokens, 74157 pages), as two 1M prompts' KV and a
#: third's. V2 hands out 78051 slots for that quota.
_ML4_1M_POOL_LENS = (1048576, 1048576, 275872)
#: The NVFP4 tp4 target at the same context ("1M tp4" cells): its delivered boot left a
#: rank 2373856 tokens (74183 pages; its 1M boot, 2352000), so its packed corner is two
#: 1M prompts' KV and a third's 276704 keys, every page of a 74183-block pool.
_ML4_1M_TP4_POOL_LENS = (1048576, 1048576, 276704)

#: The NVFP4 dep4 target at its own context (the "dep4 1M" cells): max_seq_len
#: 1048576 with chunked prefill in 32768-token steps, on V2 views of the 44-row
#: indexer pool. 36864 blocks hold a 1M prompt's 32768 pages and a mixed step's
#: decode pages (6.8 GB of indexer pool).
_DEP4_1M_V2 = dict(_ML4, layout="v2", pool_layers=44, num_blocks=36864)
#: Its packed corner, the engine-config receipt's case (d): two 5-token last
#: chunks over 983040-token prefixes on one rank pack into one indexer chunk of
#: 1966090 keys; the pool holds both requests' 61442 pages (11.4 GB; a cell peaks
#: at 24.6 GiB).
_DEP4_1M_CASE_D = dict(
    _ML4, layout="v2", pool_layers=44, num_blocks=61442, seq_lens=[5, 5], cached=[983040, 983040]
)

#: The NVFP4 tep8 target at its own context ("1M tep8" cells), on the "1M" cells' views.
#: The most keys one step's full-KV mapping holds on a tep8 rank: every token of
#: the delivered tep8 boot's pool (3680512 tokens, 115016 pages, at max_seq_len
#: 32768), as three 1M prompts' KV and a fourth's.
_ML4_1M_TEP8_POOL_LENS = (1048576, 1048576, 1048576, 534784)


class IndexerKCacheGatherOp(OpWrapper):
    """Read `num_tokens` cached indexer keys into two fresh tensors.

    Output row `i` is the `head_dim` bytes at logical flat byte offset
    `slot_mapping_fp8[k_token_start + i]` of `k_cache`, and its scale the four
    bytes at `slot_mapping_scale[k_token_start + i]`. "Logical" means the
    offset indexes `k_cache` as if it were contiguous: the kernel unravels it
    over `k_cache`'s shape and applies its real strides, so a layer view of an
    interleaved pool works, strided or not.

    Returns `(data [num_tokens, head_dim] float8_e4m3fn, scale [num_tokens, 1]
    float32)`. `k_cache` is not modified.

    Fusion boundary: a byte gather and nothing else. The caller owns the slot
    mappings, the window, and the scoring that reads the keys.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The FP8 Mistral Large 4 target's prefill (max_num_tokens 9216, max_batch_size
    # 16, no chunked prefill, no reuse) gathers a window of one step's prompt
    # keys from one of its 44 indexer layers. Its cells take that view from both
    # cache managers, pack the token budget, start the window mid-mapping, end it
    # at the mapping's end, drive every key count, and address the far end of
    # the largest pool in both layouts and at both row counts.
    #
    # The NVFP4 Mistral Large 4 target (the "32768" cells) runs the same call at
    # max_seq_len = max_num_tokens = 32768, so every one of those cells has a
    # 32768 analogue, and its weights leave a pool of up to ~95K blocks rather
    # than ~17K. The "32768, 64-token pages" cells repeat the 32768 ones over
    # 64-token pages: the same prompts and windows, pools of the same bytes in
    # half the blocks. The tep8, tp8 and dep8 targets (the "tep8/tp8/dep8"
    # cell; the indexer is whole on every rank, replicated at tep8 and tp8, data
    # parallel at dep8) make the same call, a rank holding a smaller share of
    # those weights, on a V2 pool of up to ~144K slots.
    #
    # The NVFP4 tep4 target at max_seq_len 1048576 (the "1M" cells) runs it with
    # chunked prefill: once per indexer prefill chunk, over the chunk's whole KV
    # (cached + new) in the step's full-KV mappings. Its cells gather a 1M
    # prompt's KV, start a window at key 2^20 behind one, take each query-split
    # rank's share, pack 16 chunk tails, use a real V2 manager built at 1048576,
    # address the far pool, and gather as many keys as the delivered pool holds.
    #
    # The NVFP4 tp4 target (the "1M tp4" cells) makes the same call at the same
    # context. Its cells add real V2 managers at max_seq_len 1048576 whose quota
    # is exactly their prompts' pages (one 1M prompt; a 1M prompt behind a
    # 32768-key one, gathered from key 32768) and every token of its own boot
    # pool in one window.
    #
    # The NVFP4 dep4 target at max_seq_len 1048576 (the "dep4 1M" cells) runs it
    # with chunked prefill: once per indexer prefill chunk, over the chunk's whole
    # KV (cached + new) in the step's full-KV mapping, each rank on its own
    # requests. Its cells take a 1M chunk's view from a real V2 manager built at
    # 1048576 (and a 524288 share in pre-merge), start a request's window
    # mid-mapping, put decode slots behind a 1M window, pack 16 chunk tails into
    # one window, read the 1M window's edge key counts, address the far pool, and
    # gather the packed corner past max_seq_len both whole and split per request.
    #
    # The NVFP4 tep8 target at max_seq_len 1048576 (the "1M tep8" cells) makes the
    # same call, but cuts a chunk the engine packs past max_seq_len keys at its
    # request boundaries, so one call gathers at most one request's keys, from
    # that request's offset in the full-KV mappings. Its cells take q-split ranks
    # 0 and 3's shares at eight ranks, the second request of such a packed chunk,
    # a 1M prompt at the far end of a tep8 rank's pool bound, and every rank's
    # share of each request in a mapping as long as the delivered tep8 boot's pool.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="engine V2 manager (the default), 44 rows: 3 prompts = 9216 keys, layer 71",
            spec=dict(
                _ML4,
                layout="manager_v2",
                pool_layers=44,
                layer=71,
                max_tokens=18432,
                seq_lens=[5000, 3000, 1216],
                k_token_start=0,
                num_tokens=9216,
            ),
        ),
        Cell(
            why="engine V1 manager, 44 rows: the second prompt's 3000 keys, layer 0",
            spec=dict(
                _ML4,
                layout="manager_v1",
                pool_layers=44,
                layer=0,
                max_tokens=18432,
                seq_lens=[5000, 3000, 1216],
                k_token_start=5000,
                num_tokens=3000,
            ),
        ),
        Cell(
            why="V2 view, 44 rows: 16 prompts packed to 9216 keys, one window over all",
            spec=dict(
                _ML4,
                layout="v2",
                pool_layers=44,
                pool_row=20,
                num_blocks=2048,
                seq_lens=list(_ML4_16_PROMPTS),
                k_token_start=0,
                num_tokens=9216,
            ),
        ),
        Cell(
            why="V1 view, 44 rows: 16 prompts packed to 9216 keys, one window over all",
            spec=dict(
                _ML4,
                layout="v1",
                pool_layers=44,
                pool_row=20,
                num_blocks=2048,
                seq_lens=list(_ML4_16_PROMPTS),
                k_token_start=0,
                num_tokens=9216,
            ),
        ),
        Cell(
            why="mixed step: the third of 4 prompts (2049 keys), 12 decode rows after it",
            spec=dict(
                _ML4,
                layout="v2",
                pool_layers=44,
                pool_row=43,
                num_blocks=2048,
                seq_lens=[3000, 2500, 2049, 1655],
                decode_rows=12,
                k_token_start=5500,
                num_tokens=2049,
            ),
        ),
        Cell(
            why="a 1-key window ending exactly at the mapping's end, V1 row 0",
            spec=dict(
                _ML4,
                layout="v1",
                pool_layers=44,
                pool_row=0,
                num_blocks=2048,
                seq_lens=[9215, 1],
                k_token_start=9215,
                num_tokens=1,
            ),
        ),
        Cell(
            why="every key count 1..9216 as prefix and suffix of one window (18432 calls, ~4 s)",
            spec=dict(
                _ML4,
                layout="v2",
                pool_layers=44,
                pool_row=7,
                num_blocks=2048,
                seq_lens=[9216],
                k_token_start=0,
                num_tokens=9216,
                sweep=True,
            ),
        ),
        Cell(
            why="far pool V1, 44 rows x 18432 blocks (3.4 GB): last row, bytes past 2^31",
            spec=dict(
                _ML4_FAR_44,
                layout="v1",
                pool_row=43,
                seq_lens=[9216],
                k_token_start=0,
                num_tokens=9216,
            ),
        ),
        Cell(
            why="far pool V2, 44 rows x 18432 blocks (3.4 GB): last row, slots past 2^31",
            spec=dict(
                _ML4_FAR_44,
                layout="v2",
                pool_row=43,
                seq_lens=[9216],
                k_token_start=0,
                num_tokens=9216,
            ),
        ),
        Cell(
            why="far pool V1, 72 rows x 16384 blocks (5.0 GB): last row, bytes past 2^32",
            spec=dict(
                _ML4_FAR_72,
                layout="v1",
                pool_row=71,
                seq_lens=[9216],
                k_token_start=0,
                num_tokens=9216,
            ),
        ),
        Cell(
            why="far pool V2, 72 rows x 16384 blocks (5.0 GB): last row, slots past 2^32",
            spec=dict(
                _ML4_FAR_72,
                layout="v2",
                pool_row=71,
                seq_lens=[9216],
                k_token_start=0,
                num_tokens=9216,
            ),
        ),
        Cell(
            why="32768: engine V2 manager, 44 rows: one 32768-key prompt, the max_num_tokens "
            "step's whole window, layer 71",
            spec=dict(
                _ML4_32K_MANAGER,
                layout="manager_v2",
                layer=71,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768: engine V1 manager, 44 rows: the second of 3 prompts (10000 keys from "
            "key 20000), layer 0",
            spec=dict(
                _ML4_32K_MANAGER,
                layout="manager_v1",
                layer=0,
                seq_lens=[20000, 10000, 2768],
                k_token_start=20000,
                num_tokens=10000,
            ),
        ),
        Cell(
            why="32768: V2 view, 44 rows: 16 prompts (max_batch_size) packed to 32768 keys, one "
            "window over all",
            spec=dict(
                _ML4,
                layout="v2",
                pool_layers=44,
                pool_row=20,
                num_blocks=2048,
                seq_lens=list(_ML4_32K_16),
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768: V1 view, 44 rows: 16 prompts (max_batch_size) packed to 32768 keys, one "
            "window over all",
            spec=dict(
                _ML4,
                layout="v1",
                pool_layers=44,
                pool_row=20,
                num_blocks=2048,
                seq_lens=list(_ML4_32K_16),
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768: mixed step, the call site's window: 4 prompts' 32756 keys, then 12 decode "
            "rows' new-token slots at positions up to 32767",
            spec=dict(
                _ML4,
                layout="v2",
                pool_layers=44,
                pool_row=43,
                num_blocks=8192,
                seq_lens=[12000, 10000, 8000, 2756],
                decode_pos=[32767, 1, 31, 32, 33, 1023, 2048, 9216, 16384, 20000, 30000, 32766],
                k_token_start=0,
                num_tokens=32756,
            ),
        ),
        Cell(
            why="32768: a 1-key window ending exactly at the mapping's end, V1 row 0",
            spec=dict(
                _ML4,
                layout="v1",
                pool_layers=44,
                pool_row=0,
                num_blocks=2048,
                seq_lens=[32767, 1],
                k_token_start=32767,
                num_tokens=1,
            ),
        ),
        Cell(
            why="32768: every key count 1..32768 as prefix and suffix of one window (65536 "
            "calls, ~5 s)",
            spec=dict(
                _ML4,
                layout="v2",
                pool_layers=44,
                pool_row=7,
                num_blocks=2048,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
                sweep=True,
            ),
        ),
        Cell(
            why="32768: far pool V2, 44 rows x 109568 slots (20.4 GB, large): last row, past 2^34",
            spec=dict(
                _ML4_32K_FAR,
                layout="v2",
                pool_row=43,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768: far pool V1, 44 rows x 109568 slots (20.4 GB, large): last row, past 2^34",
            spec=dict(
                _ML4_32K_FAR,
                layout="v1",
                pool_row=43,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768, 64-token pages: engine V2 manager, 44 rows: one 32768-key prompt, the "
            "max_num_tokens step's whole window, layer 71",
            spec=dict(
                _ML4_32K_P64_MANAGER,
                layout="manager_v2",
                layer=71,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768, 64-token pages: engine V1 manager, 44 rows: the second of 3 prompts "
            "(10000 keys from key 20000), layer 0",
            spec=dict(
                _ML4_32K_P64_MANAGER,
                layout="manager_v1",
                layer=0,
                seq_lens=[20000, 10000, 2768],
                k_token_start=20000,
                num_tokens=10000,
            ),
        ),
        Cell(
            why="32768, 64-token pages: V2 view, 44 rows: 16 prompts (max_batch_size) packed to "
            "32768 keys, one window over all",
            spec=dict(
                _ML4_64,
                layout="v2",
                pool_layers=44,
                pool_row=20,
                num_blocks=1024,
                seq_lens=list(_ML4_32K_16),
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768, 64-token pages: V1 view, 44 rows: 16 prompts (max_batch_size) packed to "
            "32768 keys, one window over all",
            spec=dict(
                _ML4_64,
                layout="v1",
                pool_layers=44,
                pool_row=20,
                num_blocks=1024,
                seq_lens=list(_ML4_32K_16),
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768, 64-token pages: mixed step, the call site's window: 4 prompts' 32756 "
            "keys, then 12 decode rows' new-token slots at positions up to 32767",
            spec=dict(
                _ML4_64,
                layout="v2",
                pool_layers=44,
                pool_row=43,
                num_blocks=4096,
                seq_lens=[12000, 10000, 8000, 2756],
                decode_pos=[32767, 1, 31, 32, 33, 1023, 2048, 9216, 16384, 20000, 30000, 32766],
                k_token_start=0,
                num_tokens=32756,
            ),
        ),
        Cell(
            why="32768, 64-token pages: a 1-key window ending exactly at the mapping's end, V1 "
            "row 0",
            spec=dict(
                _ML4_64,
                layout="v1",
                pool_layers=44,
                pool_row=0,
                num_blocks=1024,
                seq_lens=[32767, 1],
                k_token_start=32767,
                num_tokens=1,
            ),
        ),
        Cell(
            why="32768, 64-token pages: every key count 1..32768 as prefix and suffix of one "
            "window (65536 calls, ~5 s)",
            spec=dict(
                _ML4_64,
                layout="v2",
                pool_layers=44,
                pool_row=7,
                num_blocks=1024,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
                sweep=True,
            ),
        ),
        Cell(
            why="32768, 64-token pages: far pool V2, 44 rows x 54784 slots (20.4 GB, large): last "
            "row, past 2^34 bytes, view pages below 2410496 (0.11% of 2^31)",
            spec=dict(
                _ML4_32K_P64_FAR,
                layout="v2",
                pool_row=43,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="32768, 64-token pages: far pool V1, 44 rows x 54784 slots (20.4 GB, large): last "
            "row, past 2^34 bytes",
            spec=dict(
                _ML4_32K_P64_FAR,
                layout="v1",
                pool_row=43,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="tep8/tp8/dep8: a rank's largest V2 pool, 44 rows x 147456 slots (27.4 GB, 54.2 "
            "GiB peak, large): the 32768-key window from the last row's top 1024 blocks, to the "
            "view's last byte",
            spec=dict(
                _ML4_32K_8GPU_FAR,
                layout="v2",
                pool_row=43,
                seq_lens=[32768],
                k_token_start=0,
                num_tokens=32768,
            ),
        ),
        Cell(
            why="1M: one chunk's whole KV, a 1048576-key prompt's last chunk (1015808 cached + "
            "32768 new), V2 row 43 of 32800 blocks (6.1 GB)",
            spec=dict(
                _ML4_1M_V2,
                pool_row=43,
                num_blocks=32800,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1048576,
            ),
        ),
        Cell(
            why="1M: the second of two 1M prompts' chunks, its whole KV from key 1048576 = 2^20, "
            "V2 row 20 of 65568 blocks (12.2 GB; 26.2 GiB peak, large)",
            spec=dict(
                _ML4_1M_V2,
                pool_row=20,
                num_blocks=65568,
                seq_lens=[1048576, 1048576],
                k_token_start=1048576,
                num_tokens=1048576,
                large=True,
            ),
        ),
        Cell(
            why="1M: q-split rank 0's share of a 1M prompt's last chunk: 1015808 cached + 8192 = "
            "1024000 keys, V2 row 7",
            spec=dict(
                _ML4_1M_V2,
                pool_row=7,
                num_blocks=32800,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1024000,
            ),
        ),
        Cell(
            why="1M: a 2049-key prompt behind a 1M one, from key 1048576, V2 row 0 (its 32833 "
            "pages cycled over 32800 blocks)",
            spec=dict(
                _ML4_1M_V2,
                pool_row=0,
                num_blocks=32800,
                cycle=True,
                seq_lens=[1048576, 2049],
                k_token_start=1048576,
                num_tokens=2049,
            ),
        ),
        Cell(
            why="1M: 16 packed chunk tails' whole KV, 16 x 61456 = 983296 keys in one window, "
            "V2 row 31",
            spec=dict(
                _ML4_1M_V2,
                pool_row=31,
                num_blocks=30768,
                seq_lens=[61456] * 16,
                k_token_start=0,
                num_tokens=983296,
            ),
        ),
        Cell(
            why="1M: engine V2 manager at max_seq_len 1048576 (quota 2097152 tokens; 112.8 GiB on "
            "device, large): one 1M prompt's whole KV, layer 71",
            spec=dict(
                _ML4,
                layout="manager_v2",
                pool_layers=44,
                layer=71,
                max_tokens=2097152,
                max_seq_len=1048576,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1048576,
                large=True,
            ),
        ),
        Cell(
            why="1M: far pool V2, 44 rows x 109568 slots (20.4 GB; 42.4 GiB peak, large): a 1M "
            "prompt's whole KV in the last 32768 blocks, past 2^34",
            spec=dict(
                _ML4_32K_FAR,
                layout="v2",
                pool_row=43,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1048576,
            ),
        ),
        Cell(
            why="1M packed corner: as many keys as the boot pool holds (2373024) in one window, "
            "V2 row 43 of 78051 slots (14.5 GB; 30.7 GiB peak, large)",
            spec=dict(
                _ML4_1M_V2,
                pool_row=43,
                num_blocks=78051,
                tail=True,
                seq_lens=list(_ML4_1M_POOL_LENS),
                k_token_start=0,
                num_tokens=2373024,
                large=True,
            ),
        ),
        Cell(
            why="1M packed corner on 4096 blocks, pages cycled: 2373024 keys in one window",
            spec=dict(
                _ML4_1M_V2,
                pool_row=43,
                num_blocks=4096,
                cycle=True,
                seq_lens=list(_ML4_1M_POOL_LENS),
                k_token_start=0,
                num_tokens=2373024,
            ),
        ),
        Cell(
            why="1M windows: q-split shares and 2^k - 1, 2^k, 2^k + 1 keys to 2^21 + 1 over a "
            "2373024-key mapping, pages cycled",
            spec=dict(
                _ML4_1M_V2,
                pool_row=20,
                num_blocks=4096,
                cycle=True,
                seq_lens=list(_ML4_1M_POOL_LENS),
                k_token_start=0,
                num_tokens=2373024,
                sweep="windows",
            ),
        ),
        Cell(
            why="1M tp4: engine V2 manager at a 1048576-token quota (32768 blocks, 34487 slots; "
            "56.7 GiB device peak, large): one 1M prompt's whole KV, layer 71",
            spec=dict(
                _ML4,
                layout="manager_v2",
                pool_layers=44,
                layer=71,
                max_tokens=1048576,
                max_seq_len=1048576,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1048576,
                large=True,
            ),
        ),
        Cell(
            why="1M tp4: engine V2 manager at a 1081344-token quota (33792 blocks, 35566 slots; "
            "58.4 GiB device peak, large): a 1M prompt behind a 32768-key one, its whole KV from "
            "key 32768, layer 0",
            spec=dict(
                _ML4,
                layout="manager_v2",
                pool_layers=44,
                layer=0,
                max_tokens=1081344,
                max_seq_len=1048576,
                seq_lens=[32768, 1048576],
                k_token_start=32768,
                num_tokens=1048576,
                large=True,
            ),
        ),
        Cell(
            why="1M tp4 packed corner: every token of the tp4 boot pool (2373856) in one window, "
            "V2 row 43 of 74183 blocks, each page once (13.8 GB; 29.4 GiB peak, large)",
            spec=dict(
                _ML4_1M_V2,
                pool_row=43,
                num_blocks=74183,
                seq_lens=list(_ML4_1M_TP4_POOL_LENS),
                k_token_start=0,
                num_tokens=2373856,
                large=True,
            ),
        ),
        Cell(
            why="1M tp4 packed corner on 4096 blocks, pages cycled: 2373856 keys in one window",
            spec=dict(
                _ML4_1M_V2,
                pool_row=43,
                num_blocks=4096,
                cycle=True,
                seq_lens=list(_ML4_1M_TP4_POOL_LENS),
                k_token_start=0,
                num_tokens=2373856,
            ),
        ),
        Cell(
            why="dep4 1M manager, a 1M prompt's last chunk: 1015808 cached + 32768 new, layer 71, "
            "engine V2 manager at max_seq_len 1048576 (quota 1050624 tokens; 56.8 GiB peak, large)",
            spec=dict(
                _ML4,
                layout="manager_v2",
                pool_layers=44,
                layer=71,
                max_tokens=1050624,
                max_seq_len=1048576,
                seq_lens=[32768],
                cached=[1015808],
                k_token_start=0,
                num_tokens=1048576,
                large=True,
            ),
        ),
        Cell(
            why="dep4 1M manager, a later request's whole KV: 1000000 cached + 30720 new from key "
            "502048, layer 0, engine V2 manager at max_seq_len 1048576 (quota 1572864 tokens; "
            "85.1 GiB peak, large)",
            spec=dict(
                _ML4,
                layout="manager_v2",
                pool_layers=44,
                layer=0,
                max_tokens=1572864,
                max_seq_len=1048576,
                seq_lens=[2048, 30720],
                cached=[500000, 1000000],
                k_token_start=502048,
                num_tokens=1030720,
                large=True,
            ),
        ),
        Cell(
            why="dep4 1M manager share at max_seq_len 524288 (quota 526336 tokens; 28.9 GiB peak): "
            "a 524288-key prompt's last chunk, 491520 cached + 32768 new, layer 71",
            spec=dict(
                _ML4,
                layout="manager_v2",
                pool_layers=44,
                layer=71,
                max_tokens=526336,
                max_seq_len=524288,
                seq_lens=[32768],
                cached=[491520],
                k_token_start=0,
                num_tokens=524288,
            ),
        ),
        Cell(
            why="dep4 1M mixed step: a 1M prompt's last chunk (1015808 cached + 32768 new), then "
            "12 decode slots at positions to 1048575, V2 row 43",
            spec=dict(
                _DEP4_1M_V2,
                pool_row=43,
                seq_lens=[32768],
                cached=[1015808],
                decode_pos=[
                    1048575,
                    1,
                    31,
                    32,
                    33,
                    2048,
                    32768,
                    65536,
                    131072,
                    262144,
                    524288,
                    1048574,
                ],
                k_token_start=0,
                num_tokens=1048576,
            ),
        ),
        Cell(
            why="dep4 1M packed chunk: 16 requests' tails, 16 x (65520 cached + 16 new) = 1048576 "
            "keys in one window, V2 row 20",
            spec=dict(
                _DEP4_1M_V2,
                pool_row=20,
                seq_lens=[16] * 16,
                cached=[65520] * 16,
                k_token_start=0,
                num_tokens=1048576,
            ),
        ),
        Cell(
            why="dep4 1M edge key counts 1, 2048, 32768, 32769, 2^k + 1 (k = 16..19), 1048575 and "
            "1048576 as the 1M window's prefix and suffix, V2 row 7",
            spec=dict(
                _DEP4_1M_V2,
                pool_row=7,
                seq_lens=[32768],
                cached=[1015808],
                k_token_start=0,
                num_tokens=1048576,
                sweep_counts=[
                    1,
                    2048,
                    32768,
                    32769,
                    65537,
                    131073,
                    262145,
                    524289,
                    1048575,
                    1048576,
                ],
            ),
        ),
        Cell(
            why="dep4 1M far pool V2, 44 rows x 109568 slots (20.4 GB; 42.4 GiB peak, large): a 1M "
            "prompt's whole KV in the last 32768 blocks, past 2^34",
            spec=dict(
                _DEP4_1M_V2,
                pool_row=43,
                num_blocks=109568,
                tail=True,
                large=True,
                seq_lens=[32768],
                cached=[1015808],
                k_token_start=0,
                num_tokens=1048576,
            ),
        ),
        Cell(
            why="dep4 1M split packed chunk, request 1 of case (d): its whole KV (983040 cached + "
            "5 new) from key 0 of the 1966090-key mapping, V2 row 0",
            spec=dict(_DEP4_1M_CASE_D, pool_row=0, k_token_start=0, num_tokens=983045),
        ),
        Cell(
            why="dep4 1M split packed chunk, request 2 of case (d): its whole KV (983040 cached + "
            "5 new) from key 983045 of the 1966090-key mapping, V2 row 43",
            spec=dict(_DEP4_1M_CASE_D, pool_row=43, k_token_start=983045, num_tokens=983045),
        ),
        Cell(
            why="dep4 1M packed chunk past max_seq_len, unsplit: case (d)'s 1966090 keys in one "
            "window, V2 row 20",
            spec=dict(_DEP4_1M_CASE_D, pool_row=20, k_token_start=0, num_tokens=1966090),
        ),
        Cell(
            why="1M tep8: q-split rank 0's share of a 1M prompt's last chunk at eight ranks: "
            "1015808 cached + 4096 = 1019904 keys, V2 row 11",
            spec=dict(
                _ML4_1M_V2,
                pool_row=11,
                num_blocks=32800,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1019904,
            ),
        ),
        Cell(
            why="1M tep8: q-split rank 3's share of a 1M prompt's last chunk at eight ranks: "
            "1015808 cached + 16384 = 1032192 keys, V2 row 30",
            spec=dict(
                _ML4_1M_V2,
                pool_row=30,
                num_blocks=32800,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1032192,
            ),
        ),
        Cell(
            why="1M tep8: a packed chunk past 1M keys cut per request, two 983045-key requests' "
            "5-token tails: the second's whole KV from key 983045, V2 row 5 of 61448 blocks",
            spec=dict(
                _ML4_1M_V2,
                pool_row=5,
                num_blocks=61448,
                seq_lens=[983045, 983045],
                k_token_start=983045,
                num_tokens=983045,
            ),
        ),
        Cell(
            why="1M tep8: far pool V2, 44 rows x 147456 slots (27.4 GB, large): a 1M prompt's "
            "whole KV in the last 32768 blocks, to the view's last byte",
            spec=dict(
                _ML4_32K_8GPU_FAR,
                layout="v2",
                pool_row=43,
                seq_lens=[1048576],
                k_token_start=0,
                num_tokens=1048576,
            ),
        ),
        Cell(
            why="1M tep8 windows: eight ranks' q-split shares of every request, and 2^k - 1, 2^k, "
            "2^k + 1 keys over a 3680512-key mapping, pages cycled",
            spec=dict(
                _ML4_1M_V2,
                pool_row=37,
                num_blocks=4096,
                cycle=True,
                seq_lens=list(_ML4_1M_TEP8_POOL_LENS),
                k_token_start=2097152,
                num_tokens=1048576,
                sweep="windows",
                qsplit_ranks=8,
            ),
        ),
    )

    note = """
    Metadata consumed: none. The op reads `k_cache`, the two slot mappings and
    the three integers. A caller builds the mappings the way the DSA indexer
    does (`_compute_slot_mappings` in
    `_torch/attention/backends/sparse/dsa/indexer.py`): page = block id times
    the manager's `indexer_k_cache_page_scale`, a token's data at
    `page * tpb * 132 + (pos % tpb) * 128`, its scale at
    `page * tpb * 132 + tpb * 128 + (pos % tpb) * 4` for `tpb` tokens per
    page (4224 and 4096 at 32, 8448 and 8192 at 64).

    The views `get_indexer_k_cache_buffers` hands out at Mistral Large 4's geometry
    (page 32, 132 bytes per token, R indexer rows), measured on both managers:
      DSACacheManagerV2 (the engine default): contiguous
        [num_blocks * R - row, 32, 1, 132], starting at row * 4224 bytes into
        the indexer pool; page scale R.
      DSACacheManager (V1): [num_blocks, 32, 1, 132] with stride(0) R * 4224,
        starting at row * 4224; page scale 1.
    R is 44 when the engine's pretrained config carries `index_topk_pattern`.
    It is 72 when it does not, because `_is_full_indexer_layer` does not read the
    checkpoint's own spelling, `index_sharing`. Both are cells; the 72-row ones
    stop at the FP8 target's pool (16384 blocks). The NVFP4 target's engine
    config, `MistralLarge4Config`, derives `index_topk_pattern` from
    `index_sharing`, so it runs at R = 44. A request's blocks take
    descending slots on V2 and ascending ones on V1; the op reads whatever the
    mappings name. At the NVFP4 target's max_seq_len = max_num_tokens = 32768,
    a V2 quota of 65536 tokens gave 2155 slots (2048 blocks of quota), and the
    views of layers 0 and 71 were [94820, ...] and [94777, ...], as above.
    The 64-token manager cells gather through both managers' views at 64
    tokens per page, built for the same 32768-token limits.
    At tep8 and tp8 a rank holds 73.61 GiB of weights and at dep8 89.29 GiB
    (computed from the checkpoint's shard headers), which leave room for about
    144075 and 132949 V2 slots; no boot has measured those pools. Views are
    certified to 147456 slots on V2 (the eight-GPU cell) and to 109568 on V1;
    past those they are not certified.

    Call site (the Mistral Large 4 targets' `PrefillTarget._indexer`): one call per
    indexer layer of a context step whose longest prompt passes index_topk
    2048, on that layer's view and the step's slot mappings. With chunked
    prefill and reuse off the engine plans no indexer chunk at these sizes
    (`indexer_max_chunk_size` defaults to 32768), so `k_token_start` is 0 and
    `num_tokens` is every context key of the step, up to 32768; a mixed step's
    decode slots follow them in the mapping.

    With chunked prefill on (the NVFP4 tep4, tp4 and dep4 targets at
    max_seq_len 1048576, max_num_tokens 32768) every chunk after a request's
    first has a cached prefix. `Indexer.prepare_for_chunked_prefill` then plans
    the indexer's chunks with `plan_mla_chunked_prefill_chunks`, and once any
    context request of the step has cached keys,
    `recompute_context_kv_gather_mappings` builds the full-KV mappings
    (`slot_mapping_fp8_fullkv` / `slot_mapping_scale_fullkv`): every context
    request's cached and new keys, request after request, and no decode slots
    (a step whose contexts have no cached keys keeps the regular mappings,
    decode slots after the context keys). The target calls the op once per
    indexer layer and chunk, from the chunk's `k_token_start` (its first
    request's offset in that concatenation) over its
    `k_token_end - k_token_start` keys:
      one request: its cached + new keys, up to 1048576 (the 1M and dep4 1M
        cells);
      a group packed while (sum of new) x (sum of KV) stays within the 2^28
        packed-logits budget: every key of the group, which the pool bounds,
        not max_seq_len (two 1M prompts' short tails pack into one chunk); the
        cells reach 2373024 keys, every token of the delivered tep4 boot's pool,
        and 2373856, every token of the delivered tp4 boot's (74183 pages; the
        tp4 boot at max_seq_len 1048576 left 2352000). Two 5-token last chunks
        over 983040-token prefixes pack into one chunk of 1966090 keys (the
        dep4 boot's pool holds 1983872 tokens per rank); split at the request
        boundary, as the dep4 target's indexer may do, it is two calls,
        (0, 983045) and (983045, 983045);
      a request cut by a rank's share under the engine's query split: from its
        offset to its cached keys plus the share's end, cached + 8192 (r + 1)
        keys for rank r of a 32768-row chunk.
    Under attention DP (the dep4 target) the engine does not split the
    indexer's prefill queries across ranks, so a rank's calls cover its own
    requests whole.
    The 1M cells' V2 views are rebuilt as above, but one: a real V2 manager at
    max_seq_len 1048576 with a quota of 2097152 tokens kept max_seq_len
    1048576, handed out a [1, 32772] block table and a layer-71 view of
    [3034813, 32, 1, 132] (68974 slots), and put a 1048576-key prompt's 32768
    pages in slots 0..32767 (data bytes 0..6089947520 of the view). So did a
    quota of exactly the prompt's pages, 1048576 tokens (32768 blocks): V2
    kept max_seq_len 1048576 and the [1, 32772] table, handed out 34487
    slots (layer 71's view [1517385, 32, 1, 132]) and took the prompt in
    slots 32767..0. A 1081344-token quota (33792 blocks) took a
    32768-token prompt in slots 1023..0 and a 1M prompt behind it in slots
    33791..1024, again exactly the quota, with 35566 slots (layer 0's view
    [1564904, 32, 1, 132]); a window from key 32768 gathers the second. The
    "cycled" cells reach these mapping lengths on a small pool: their pages
    outnumber the pool and are drawn as consecutive fresh permutations of it;
    a window started one key early, or wrapped at key 2^20, reads other bytes
    there (a test proves it).
    For the "dep4 1M" cells, a real V2 manager at max_seq_len 1048576 with a
    quota of 1050624 tokens kept max_seq_len 1048576, handed out 34563 slots
    (49.4 GiB on device), a [1, 32772] block table and a layer-71 view of
    [1520729, 32, 1, 132], and put a 1048576-key prompt's 32768 pages in
    slots 32767..0, descending; a quota of 1572864 tokens gave 51730 slots
    (73.6 GiB) and put a 502048-key and a 1030720-key prompt in slots 15688..0
    and 47898..15689. A quota too small to hold max_seq_len tokens lowers the
    manager's max_seq_len to what it holds, so the pre-merge "dep4 1M" cells
    rebuild V2 views instead, and a share runs a real manager at max_seq_len
    524288 (quota 526336 tokens: 17319 slots, a [1, 16388] table, 25.1 GiB).
    In the "dep4 1M" mixed-step cell the decode slots stand behind a 1M window
    as in the regular mapping.

    The NVFP4 tep8 target at max_seq_len 1048576 makes tep4's calls but one:
    a chunk the engine packs past max_seq_len keys is cut at its request
    boundaries (`_per_request_chunks` in the target), one call per request over
    that request's own keys from its offset in the full-KV mappings, so a call
    gathers at most 1048576 keys. Two 983045-key requests' 5-token tails make
    one chunk of 1966090 keys, gathered as (0, 983045) and (983045, 983045).
    The offsets still reach the pool's size: the "1M tep8" windows run over a
    3680512-key mapping, every token of the delivered tep8 boot's pool. Under
    the engine's query split over the eight ranks, rank r's chunk of a
    request's last 32768 rows stops at cached + 4096 (r + 1) keys
    (`_split_prefill_queries`, `plan_mla_chunked_prefill_chunks` and
    `Indexer.prepare_one_prefill_chunk` in `dsa/indexer.py`; a test checks the
    shares and the packed chunk against them). Device peaks: 13.4 GiB for the
    shares on 32800 blocks, 24.6 GiB for the packed chunk's 61448 blocks, 56.3
    GiB for the 147456-slot far view (`large`).

    Preconditions the op enforces itself:
      k_cache uint8, 4-D, dim 2 == 1;
      both mappings int64, 1-D and contiguous;
      every tensor on the device;
      k_token_start + num_tokens within both mappings.
    It raises on each, with these messages:
      "expected scalar type Byte but found Float8_e4m3fn"
      "k_cache must be a 4D Tensor [num_blocks, block_size, 1, per_token_size], got %d dimensions3"
      "k_cache dimension 2 must be 1 for DeepSeek-V3.2, got %d2"
      "slot_mapping_fp8 must be int64"
      "slot_mapping_fp8 must be a 1D Tensor"
      "slot_mapping_fp8 must be contiguous"
      "All tensors must be CUDA tensors"
      "slot_mapping_fp8 too short for k_token_start + num_tokens"
      "indexer_k_cache_gather_op head_dim must be 128 (FP8) or 64 (FP4 packed), got %d96"
    The literal "%d" is in the engine's text.

    Alignment: every slot must be a multiple of 4 and the view's data pointer
    4-byte aligned. A misaligned data slot, scale slot or base pointer faults
    the CUDA context, which is not recoverable in that process:
    "torch.AcceleratorError: CUDA error: misaligned address".

    `num_tokens = 0` checks `head_dim`, then returns a (0, head_dim)
    float8_e4m3fn and a (0, 1) float32 tensor without checking or reading any
    other argument.

    The outputs are fresh, contiguous tensors. No CUDA-graph capture is
    certified: the call site runs in eager context steps.

    Cost: a call is launch-bound at every key count from 1 to 32768, on V1
    and V2 views and at the far ends of the 109568-slot pool and the
    147456-slot V2 view: 32768 keys are 4.1 MiB.
    Past that the time grows with the key count (the grid is one 32-thread
    block per key).

    Memory: a call allocates its two outputs and nothing else,
    num_tokens * (head_dim + 4) bytes. At num_tokens 1048576 that is 134217728
    bytes of keys and 4194304 of scales, 138412032 bytes (132 MiB): torch's
    allocated peak over one such call rose by exactly 138412032 bytes.

    The FP4 cache (head_dim 64, 68-byte rows) uses the same kernel with half the
    bytes per token. No target uses it, and it is not certified.

    Hard limits (sm_103)
      slot           int64 logical byte offset, unravelled over k_cache's
                     shape, then multiplied by the int64 strides
                     boundary 2^31 / 2^32 bytes
                     probed at 4982833151 (72 rows x 16384 blocks, V1 and V2)
                     and at 20363870207 (44 rows x 109568 slots, the NVFP4
                     tep4 target's largest pool, V1 and V2: past 2^34), there
                     also with a 1M prompt's 1048576 keys (V2, dep4 1M), the
                     same bytes at 64-token pages (54784 slots, V1 and V2),
                     and at slot 27405400700, the last scale of a 147456-slot
                     V2 view of 27405400704 bytes (44 rows, row 43, a tep8,
                     tp8 or dep8 rank's largest pool: past 2^34, below 2^35)
                     passed
      page index     int32 after the unravel; k_cache.size(0) is also cast
                     to int32
                     boundary 2^31 pages
                     largest certified view 6488021 pages (V2, 44 rows x
                     147456 slots); a whole 276.62 GiB GB300 holds 70317459
                     pages of 4224 bytes: not reachable
      window index   k_token_start + i in int32 (the op narrows k_token_start
                     and num_tokens to int32 before the launch)
                     boundary 2^31
                     mapping <= 32768 entries without chunked prefill; with
                     it, every context key of the step, which the pool
                     bounds (1983872 tokens at the dep4 boot): certified to
                     3680512 entries (k_token_start + i up to 3680511, the
                     tep8 windows), on tp4's packed corner to 2373856
                     (k_token_start + i up to 2373855), and on dep4's packed
                     corner to 1966090 entries (k_token_start up to 983045,
                     k_token_start + i up to 1966089); not reached
      num_tokens     int32 grid, one block per key; output offsets row * 128
                     in int64
                     boundary 2^31
                     32768 without chunked prefill; certified to 3680512
                     with it (a tep8 call gathers at most 1048576), to
                     2373856 on tp4's packed corner (a grid of 2373856
                     blocks, 303853568 output bytes), on dep4's packed corner
                     to 1966090; not reached
      fp32           no index passes through fp32
    """

    def __call__(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.indexer_k_cache_gather_op(
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            k_token_start,
            num_tokens,
            head_dim,
        )

    def reference(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A byte gather over the cache's logical (contiguous) order."""
        flat = k_cache.contiguous().view(torch.uint8).reshape(-1)
        window = slice(k_token_start, k_token_start + num_tokens)
        data_at = slot_mapping_fp8[window].long().unsqueeze(1)
        scale_at = slot_mapping_scale[window].long().unsqueeze(1)
        data = flat[data_at + torch.arange(head_dim, device=flat.device)]
        scale = flat[scale_at + torch.arange(4, device=flat.device)]
        return (
            data.view(torch.float8_e4m3fn),
            scale.contiguous().view(torch.float32).reshape(num_tokens, 1),
        )

    def is_valid(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> None:
        if num_tokens == 0:
            return  # two empty tensors, and nothing is read
        # The op checks only the upper end of the window. A negative start reads the
        # int64 words in front of the mapping and uses them as slots.
        assert k_token_start >= 0, f"k_token_start must be >= 0, got {k_token_start}"
        # Width and range are checked only where the op does not refuse loudly itself.
        cache_ok = k_cache.dtype == torch.uint8 and k_cache.dim() == 4
        if cache_ok and head_dim in (64, 128):
            # head_dim is the gather width, and the op admits either width on any
            # cache: 64 on a 132-byte row returns the first half of every key.
            assert k_cache.shape[-1] == head_dim + 4, (
                f"head_dim {head_dim} does not match the cache's {k_cache.shape[-1]}-byte rows "
                f"(head_dim key bytes plus a 4-byte scale)"
            )
        window = slice(k_token_start, k_token_start + num_tokens)
        data_at, scale_at = slot_mapping_fp8[window], slot_mapping_scale[window]
        # The op refuses these itself: a window past a mapping's end, a mapping that
        # is not 1-D int64, mappings on two devices.
        if (
            data_at.shape != (num_tokens,)
            or scale_at.shape != (num_tokens,)
            or data_at.dtype != torch.int64
            or scale_at.dtype != torch.int64
            or data_at.device != scale_at.device
        ):
            return
        # Both windows' bounds in one host read: a test runs this before every call.
        lo, hi = torch.aminmax(torch.stack([data_at, scale_at]), dim=1)
        lo_data, lo_scale, hi_data, hi_scale = torch.cat([lo, hi]).tolist()
        # A negative slot makes the kernel skip that row's data and scale, and the
        # output is torch.empty: the row comes back as whatever the allocator held.
        assert lo_data >= 0 and lo_scale >= 0, (
            "every slot in the window must be >= 0; a negative slot leaves its row uninitialized"
        )
        # Nothing bounds a slot from above: one past the view reads the memory after it.
        if cache_ok:
            extent = k_cache.numel()
            assert hi_data <= extent - head_dim and hi_scale <= extent - 4, (
                f"every slot in the window must address bytes inside k_cache's {extent} bytes"
            )


indexer_k_cache_gather_op = IndexerKCacheGatherOp()
