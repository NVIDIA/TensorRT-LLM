# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather one prefill chunk's DSA indexer keys back out of the indexer K cache."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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

#: The NVFP4 tep4 deployment at its own context ("1M" cells): max_seq_len 1048576 with
#: chunked prefill in 32768-token steps, on V2 views of the 44-row indexer pool.
_ML4_1M_V2 = dict(_ML4, layout="v2", pool_layers=44)
#: The most keys one step's full-KV mapping holds: every token of the delivered
#: tep4 boot's pool (2373024 tokens, 74157 pages), as two 1M prompts' KV and a
#: third's. V2 hands out 78051 slots for that quota.
_ML4_1M_POOL_LENS = (1048576, 1048576, 275872)
#: The NVFP4 tp4 deployment at the same context ("1M tp4" cells): its delivered boot left a
#: rank 2373856 tokens (74183 pages; its 1M boot, 2352000), so its packed corner is two
#: 1M prompts' KV and a third's 276704 keys, every page of a 74183-block pool.
_ML4_1M_TP4_POOL_LENS = (1048576, 1048576, 276704)

#: The NVFP4 dep4 deployment at its own context (the "dep4 1M" cells): max_seq_len
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

#: The NVFP4 tep8 deployment at its own context ("1M tep8" cells), on the "1M" cells' views.
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
    # half the blocks. The tep8, tp8 and dep8 deployments (the "tep8/tp8/dep8"
    # cell; the indexer is whole on every rank, replicated at tep8 and tp8, data
    # parallel at dep8) make the same call, a rank holding a smaller share of
    # those weights, on a V2 pool of up to ~144K slots.
    #
    # The NVFP4 tep4 deployment at max_seq_len 1048576 (the "1M" cells) runs it with
    # chunked prefill: once per indexer prefill chunk, over the chunk's whole KV
    # (cached + new) in the step's full-KV mappings. Its cells gather a 1M
    # prompt's KV, start a window at key 2^20 behind one, take each query-split
    # rank's share, pack 16 chunk tails, use a real V2 manager built at 1048576,
    # address the far pool, and gather as many keys as the delivered pool holds.
    #
    # The NVFP4 tp4 deployment (the "1M tp4" cells) makes the same call at the same
    # context. Its cells add real V2 managers at max_seq_len 1048576 whose quota
    # is exactly their prompts' pages (one 1M prompt; a 1M prompt behind a
    # 32768-key one, gathered from key 32768) and every token of its own boot
    # pool in one window.
    #
    # The NVFP4 dep4 deployment at max_seq_len 1048576 (the "dep4 1M" cells) runs it
    # with chunked prefill: once per indexer prefill chunk, over the chunk's whole
    # KV (cached + new) in the step's full-KV mapping, each rank on its own
    # requests. Its cells take a 1M chunk's view from a real V2 manager built at
    # 1048576 (and a 524288 share in pre-merge), start a request's window
    # mid-mapping, put decode slots behind a 1M window, pack 16 chunk tails into
    # one window, read the 1M window's edge key counts, address the far pool, and
    # gather the packed corner past max_seq_len both whole and split per request.
    #
    # The NVFP4 tep8 deployment at max_seq_len 1048576 (the "1M tep8" cells) makes the
    # same call, but cuts a chunk the engine packs past max_seq_len keys at its
    # request boundaries, so one call gathers at most one request's keys, from
    # that request's offset in the full-KV mappings. Its cells take q-split ranks
    # 0 and 3's shares at eight ranks, the second request of such a packed chunk,
    # a 1M prompt at the far end of a tep8 rank's pool bound, and every rank's
    # share of each request in a mapping as long as the delivered tep8 boot's pool.
    CELLS: tuple[Cell, ...] = (
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
    )

    note = """
    Gathers paged FP8 indexer K-cache rows (codes + scales) into contiguous windows for
    prefill MQA scoring. Covers 32K and 1M chunked/q-split windows, mixed steps, and far pools.
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
