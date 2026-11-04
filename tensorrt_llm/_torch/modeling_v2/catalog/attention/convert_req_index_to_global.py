# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a DSA request-local KV position to its row in the paged latent KV pool."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: Mistral Large 4 FP8 at TEP4 under its serving envelope (max_seq_len = max_num_tokens = 9216,
#: max_batch_size 16): pages of 32 tokens, 2048 selected keys, a pool that interleaves all
#: 72 layers' pages (`stride_factor` 72 x 32 = 2304), and the cache manager's block table of
#: max_batch_size rows by 9216 / 32 = 288 columns. Page ids are drawn from an 18432-block
#: pool, more than a rank can hold: about 24.7 GiB is left per GB300 after weights and a
#: token costs about 47 KB (latent plus indexer keys), which is about 17.5K pages -- an
#: estimate, as the pool has not been measured yet. Rows of the table no request reaches
#: hold stale page ids rather than -1, as the engine leaves them.
_ML4 = dict(
    block_size=32, topk=2048, pool_layers=72, table_rows=16, table_width=288, pool_blocks=18432
)

#: Mistral Large 4 NVFP4 at TEP4 under its serving envelope (max_seq_len = max_num_tokens =
#: 32768, no chunked prefill, max_batch_size 16): the same pages, top-k and 72-layer stride,
#: and the cache manager's block table of max_batch_size rows by 32768 / 32 = 1024 columns.
#: Page ids are drawn from a 107917-block pool, the most one GB300 rank could hold: the
#: device's 276.62 GiB less the target's 143.24 GiB of weights, all of it 72-layer fp8
#: latent at 72 x 32 x 576 = 1327104 bytes a page -- a bound, as no gate has measured the
#: engine's pool, which also holds indexer keys and leaves room for activations.
_ML4_32K = dict(
    block_size=32, topk=2048, pool_layers=72, table_rows=16, table_width=1024, pool_blocks=107917
)

#: The `layer_id` each of Mistral Large 4's 72 layers passes in one step: its own pool row,
#: except the KV sharers 39, 43, ..., 71, which read layer 35's latent (the checkpoint's
#: `kv_source_layers`).
_ML4_STEP = tuple(35 if i >= 39 and i % 4 == 3 else i for i in range(72))

#: The fp8_ds_mla pool (MistralLarge4CacheManager) holds pages for the 63 KV owners only:
#: stride 63 x tokens_per_block (2016 at 32-token pages), and layer i passes its source's
#: owner row -- the sharers 39, 43, ..., 71 read layer 35's, owner row 35. Both are plain
#: expressions, so the cells read without running the module: `pool_layers` 63 is
#: len(_ML4_OWNERS), and `_ML4_DS_STEP[i]` is `_ML4_OWNERS.index(_ML4_STEP[i])` in closed form
#: (an owner past 39 sits (i - 35) // 4 rows lower, one per sharer below it); the test
#: checks both against the owner list.
_ML4_OWNERS = tuple(i for i in range(72) if _ML4_STEP[i] == i)
_ML4_32K_DS = dict(_ML4_32K, pool_layers=63)
_ML4_DS_STEP = tuple(
    35 if i >= 39 and i % 4 == 3 else i - (i - 35) // 4 if i >= 39 else i for i in range(72)
)

#: The same pool at 64-token pages: stride 63 x 64 = 4032, a block table of 32768 / 64 = 512
#: columns, and 53958 blocks, the 107917-block bound's token capacity. Its last row,
#: 53957 x 4032 + 62 x 64 + 63 = 217558655, is 10.1% of 2**31, as at 32-token pages.
_ML4_32K_DS64 = dict(_ML4_32K_DS, block_size=64, table_width=512, pool_blocks=53958)

#: Mistral Large 4 NVFP4 on eight GB300s over two nodes on the fp8_ds_mla pool -- tep8 and tp8
#: (8 of the 64 heads a rank; the two shard attention alike) and dep8 (eight attention-DP
#: ranks, all 64 heads on each): the call unchanged, page ids from a 147456-block pool. A tep8
#: or tp8 rank holds 73.61 GiB of weights and a dep8 rank 89.29 GiB, which leave 203.01 and
#: 187.33 GiB of a 276.62 GiB GB300; at 47136 bytes a token (63 owner rows of 656 bytes and 44
#: indexer-key rows of 132), 1508352 bytes a 32-token page, that is 144519 and 133355 pages:
#: 147456 is above both -- a bound computed from the shard headers; the engine's pool, which
#: also leaves room for activations, is smaller.
_ML4_32K_DS_8GPU = dict(_ML4_32K_DS, pool_blocks=147456)

#: The same pool behind the block table the engine's cache manager builds
#: (`MistralLarge4CacheManager`, a `DSACacheManagerV2`): `max_blocks_per_seq` columns,
#: (32768 + 1) / 32 rounded up to 1025 and padded to a multiple of 4.
_ML4_32K_DS_8GPU_V2 = dict(_ML4_32K_DS_8GPU, table_width=1028)

#: 16 requests packed into 9216 tokens, the context step at max_batch_size.
_ML4_PACKED = [1, 7, 33, 100, 255, 257, 513, 1023, 1025, 2047, 2049, 931, 299, 301, 201, 174]

#: 16 requests packed into 32768 tokens, the NVFP4 target's context step at max_batch_size.
_ML4_32K_PACKED = [
    1,
    7,
    33,
    100,
    255,
    257,
    513,
    1023,
    1025,
    2047,
    2049,
    4095,
    4097,
    8191,
    8193,
    882,
]

#: Mistral Large 4 NVFP4 at TEP4 at the checkpoint's own context (max_seq_len 1048576, chunked
#: prefill in steps of max_num_tokens 32768, max_batch_size 16): the same pages, top-k,
#: 72-layer stride and 107917-block pool bound, and a block table of max_batch_size rows by
#: 1048576 / 32 = 32768 columns. Positions reach 1048575, column 32767.
_ML4_1M = dict(
    block_size=32, topk=2048, pool_layers=72, table_rows=16, table_width=32768, pool_blocks=107917
)

#: The width the engine's table has at 1048576: `KVCacheManagerV2` sizes `max_blocks_per_seq`
#: for max_seq_len plus one generation token of headroom, ceil(1048577 / 32) = 32769 pages,
#: padded to a multiple of 4 (`_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`), and the
#: DSA metadata's `block_table` is that wide. No position below max_seq_len reaches its last
#: four columns.
_ML4_1M_V2 = dict(_ML4_1M, table_width=32772)

#: The tp4 and tep4 targets on the fp8_ds_mla pool at 1048576: the engine's 32772-column table
#: and the 107917-block bound, at the 63 owners' stride 2016. Their boots at this envelope held
#: 69011 blocks (2208352 tokens; tp4) and 72757 blocks (2328224 tokens; tep4) per rank, inside
#: the bound.
_ML4_1M_DS = dict(_ML4_1M_V2, pool_layers=63)

#: The tep8 target on the fp8_ds_mla pool at 1048576: the engine's 32772-column table and the
#: eight-rank 147456-block bound (`_ML4_32K_DS_8GPU`'s), at the 63 owners' stride 2016. Its boot
#: at this envelope held 3658432 tokens, 114326 blocks, per rank, inside the bound.
_ML4_1M_TEP8_DS = dict(_ML4_1M_DS, pool_blocks=147456)

#: 16 chunk tails packed into one 32768-row context step, each request's query rows; its KV
#: lengths (cached prefix plus those rows) are the cells' `kv_lens`: the last token of a
#: 1048576-token prompt, fresh prompts, and tails behind prefixes of 1, 31, 32, 33, 2047,
#: 2048, 2049, 32767, 65535, 131072, 262143 and 500000 tokens.
_ML4_1M_TAIL_ROWS = [1, 7, 31, 32, 33, 100, 255, 257, 513, 1023, 1025, 2047, 2049, 8191, 8193, 9011]

#: Mistral Large 4 NVFP4 at DEP4 at the checkpoint's own context (max_seq_len 1048576, chunked
#: prefill in steps of max_num_tokens 32768 per rank, max_batch_size 16 per rank): the same
#: pages, top-k, 72-layer stride and 107917-block pool bound (a dep4 rank's V2 pool holds about
#: 62K slots), and the DSA metadata's block table of max_num_sequences 16 rows by the V2
#: manager's `max_blocks_per_seq`, 32772 columns: ceil(1048577 / 32) = 32769 pages (one
#: generation token of headroom), padded to a multiple of 4, as a `DSACacheManagerV2` built at
#: max_seq_len 1048576 reports it. Positions reach 1048575, column 32767; no position below
#: max_seq_len reaches the last four columns.
_ML4_1M_DEP4 = dict(
    block_size=32, topk=2048, pool_layers=72, table_rows=16, table_width=32772, pool_blocks=107917
)

#: The dep4 target on the fp8_ds_mla pool at 1048576: the same 16 x 32772 table and
#: 107917-block bound at the 63 owners' stride 2016. The dep4 boot on this pool held 64633
#: blocks (2068256 tokens) on every rank, inside the bound.
_ML4_1M_DEP4_DS = dict(_ML4_1M_DEP4, pool_layers=63)


class ConvertReqIndexToGlobal(OpWrapper):
    """Turn the indexer's per-request top-k positions into pool rows.

    Entry `token_indices[i, j]` is a KV position `p` counted from the start of
    request `req_id[i]`'s own sequence. The op returns

        block_table[req_id[i], p // block_size] * stride_factor
            + layer_id * block_size + p % block_size

    which is the row of the latent pool viewed as `[-1, head_dim]` that holds
    that token's latent KV for this layer. A position that cannot be resolved
    comes back as -1: `p < 0` (the top-k padding), `p // block_size` at or past
    `block_table.size(1)`, or a negative block-table entry (an unused page).

    Returns a fresh contiguous int32 tensor shaped like `token_indices`; no
    argument is modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # It is called once per layer on the step's whole top-k buffer: one row per
    # query token, `topk` columns. `pool_layers` sets `stride_factor =
    # pool_layers * block_size`, the page stride of a pool that interleaves
    # every layer's pages; `layer_id` picks the layer inside a page.
    #
    # Mistral Large 4 calls it the same way, at its envelope: decode rows are the
    # sequences of a captured batch (1, 2, 4, 8, 16, padded), each `graph` cell
    # capturing the step's 72 calls (`layer_ids`) and replaying them on new
    # batch data; prefill rows reach 9216 per call (the FP8 target, "ML4") or
    # 32768 (the NVFP4 target, "ML4 32768": DecodeTarget._attention and
    # PrefillTarget._attention, one call per layer). On eight GPUs over the
    # fp8_ds_mla pool ("ML4 fp8_ds_mla tep8/tp8/dep8", the calls of all three
    # targets) the call is the same and only the pool grows, to page ids up to
    # 147455; the "dep8 V2 table" cells add a rank's attention-DP padding (one
    # live sequence replaying the batch-16 graph behind 15 dummies) and a packed
    # context step over the 1028-column table. `kv_lens` are the requests' KV
    # lengths and `seq_rows` their query rows, so row `j` of request `s` sees
    # positions `[0, kv_lens[s] - seq_rows[s] + j + 1)`; `last_position` puts
    # each row's newest position among its selections, so a context that fills
    # the table reads its last column.
    #
    # The NVFP4 target at max_seq_len 1048576 ("ML4 1M") keeps those call sites
    # and steps of at most 32768 rows; what grows is the position (to 1048575)
    # and the table (32768 columns, and the 32772 the engine allocates). A
    # chunk's rows sit behind the request's cached prefix, so a step packs chunk
    # tails whose positions reach far past its own row count.
    #
    # The tp4 target ("ML4 1M tp4") makes the same calls at the same envelope:
    # `md.block_table`, the stride of the pool mapping, the step's index-mode
    # context rows and a mixed step's generation rows from
    # `PrefillTarget._attention`, and the step's 72 calls from
    # `DecodeTarget._attention`. `layer_ids="all"` is every layer id of the page,
    # 0..pool_layers - 1, with one call each. `order="chunk"` splits each chunk
    # row's selections between the chunk's own rows up to that row (half of
    # top-k behind a cached prefix of half or more, all of them when there are
    # fewer, more behind a shorter prefix) and the request's cached prefix (the
    # rest). Each part is drawn by strata, and the columns come in random order.
    #
    # The tp4 and tep4 targets on the fp8_ds_mla pool ("ML4 1M fp8_ds_mla" and
    # "ML4 1M tep4 fp8_ds_mla") pass stride 63 x 32 = 2016 and each layer's
    # `latent_pool_row(kv_source)` (`_ML4_DS_STEP`) at three sites: the step's 72
    # calls from `DecodeTarget._attention`; a mixed step's generation rows from
    # `PrefillTarget._attention`, whose requests are the table's rows from
    # `num_contexts` on (`first_seq`); and, on a step whose context KV exceeds
    # 2**19 rows, every context row of the step in one call from
    # `PrefillTarget._attn_ctx`. `layer_ids="owner_rows_0_62"` is every owner row
    # of the 63-owner page, one call each.
    #
    # The NVFP4 dep4 target at max_seq_len 1048576 ("dep4 1M") keeps those call
    # sites, each rank running its own requests, and steps of at most 32768
    # rows; what grows is the position (to 1048575) and the table (the 32772
    # columns the engine allocates). A chunk's rows sit behind the request's
    # cached prefix, so a step's positions reach far past its own row count.
    # The engine captures each decode graph at boot on its longest dummy
    # request, max_seq_len - 1 = 1048575 keys, beside 1-key padding rows, and
    # replays it on the step's data: `replay_kv_lens` names the replays' contexts
    # (without it a graph cell replays the capture's contexts reversed).
    #
    # The dep4 target on the fp8_ds_mla pool ("dep4 1M fp8_ds_mla") passes stride
    # 63 x 32 = 2016 and each layer's `latent_pool_row(kv_source)` (`_ML4_DS_STEP`)
    # at three sites: the step's 72 calls from `DecodeTarget._attention`; a mixed
    # step's generation rows from `PrefillTarget._attention`, whose requests are
    # the table's rows from `num_contexts` on (`first_seq`); and every context row
    # of a step whose context KV exceeds 2**19 rows in one call, from
    # `PrefillTarget._attn_ctx` ("ctx rows"). `boot_capture` captures a graph on
    # the boot's warm-up batch, the longest dummy (max_seq_len - 1 keys) first and
    # one-key fillers, each a request of its own. Under attention DP a rank pads
    # its live rows to the largest rank's graph batch with one CUDA-graph padding
    # dummy repeated (one key, one page): `pad` (and per replay, `replay_pad`)
    # makes the last that many requests that dummy, rows of the table of their own
    # holding the same page. `order="chunk"` is the split above.
    # `layer_ids="owner_rows_0_62"` is every owner row of the 63-owner page.
    #
    # The tep8 target on the fp8_ds_mla pool passes stride 63 x 32 = 2016 and
    # each layer's `latent_pool_row(kv_source)` (`_ML4_DS_STEP`). At 32768 its
    # calls are the "ML4 fp8_ds_mla tep8/tp8/dep8" cells': only its pool grows,
    # to page ids up to 147455. At max_seq_len 1048576 ("ML4 1M tep8
    # fp8_ds_mla", `_ML4_1M_TEP8_DS`) it calls the op at the three sites tp4's
    # and tep4's calls do: the step's 72 calls from `DecodeTarget._attention`;
    # a mixed step's generation rows from `PrefillTarget._attention`, whose
    # requests are the table's rows from `num_contexts` on (`first_seq`); and,
    # on a step whose context KV exceeds 2**19 rows, every context row of the
    # step in one call from `PrefillTarget._attn_ctx`. Its cells are tep4's on
    # the 147456-block bound. `layer_ids="owner_rows_0_62"` is every owner row
    # of the 63-owner page, one call each; `order="chunk"` is the split above.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="ML4 decode graph batch 1 (captured), 72 calls: one sequence at "
            "max_seq_len 9216, its last position in the table's last column",
            spec=dict(_ML4, graph=True, seq_rows=[1], kv_lens=[9216], layer_ids=_ML4_STEP),
        ),
        Cell(
            why="ML4 decode graph batch 2 (captured), 72 calls: contexts of 1 and "
            "exactly 2048 in the dense causal order the engine writes for a skipped indexer",
            spec=dict(
                _ML4,
                graph=True,
                order="causal",
                seq_rows=[1, 1],
                kv_lens=[1, 2048],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 decode graph batch 4 (captured), 72 calls: contexts of 2049 "
            "(one position past top-k), 31 and 33 (one page and a page plus one), 5000",
            spec=dict(
                _ML4,
                graph=True,
                seq_rows=[1] * 4,
                kv_lens=[2049, 31, 5000, 33],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 decode graph batch 8 (captured), 72 calls: contexts across "
            "1..9216, several longer than top-k",
            spec=dict(
                _ML4,
                graph=True,
                seq_rows=[1] * 8,
                kv_lens=[9216, 4097, 2047, 64, 100, 8000, 3000, 6145],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 decode graph batch 16 (captured), 72 calls: 13 sequences and 3 "
            "padding rows (1-token dummies), pages at the far end of the pool",
            spec=dict(
                _ML4,
                graph=True,
                far_end=True,
                seq_rows=[1] * 16,
                kv_lens=[9216, 1, 2048, 2049, 512, 7777, 33, 9000, 4096, 1500, 300, 8191, 6000]
                + [1] * 3,
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 prefill at max_num_tokens: one 9216-token request, causal windows, "
            "the last pool layer; every row count 1..9216 is a bitwise prefix of it (about 1.4 s)",
            spec=dict(_ML4, seq_rows=[9216], kv_lens=[9216], layer_ids=(71,), prefix_sweep=True),
        ),
        Cell(
            why="ML4 context step at max_batch_size: 16 requests packed into 9216 rows, "
            "the KV sharers' pool row 35, pages at the far end of the pool",
            spec=dict(
                _ML4, far_end=True, seq_rows=_ML4_PACKED, kv_lens=_ML4_PACKED, layer_ids=(35,)
            ),
        ),
        Cell(
            why="ML4 mixed step: 3 context requests, then 13 generation rows with "
            "contexts up to 9216, in one 6013-row call",
            spec=dict(
                _ML4,
                seq_rows=[3000, 2000, 1000] + [1] * 13,
                kv_lens=[3000, 2000, 1000]
                + [9216, 2, 2048, 2049, 4500, 8000, 1, 700, 3333, 9100, 6000, 129, 5000],
                layer_ids=(0,),
            ),
        ),
        Cell(
            why="ML4 every layer id 0..71 at stride 72 x 32 = 2304, eager, 16 decode "
            "rows: a pool that gives the KV sharers rows of their own",
            spec=dict(
                _ML4,
                seq_rows=[1] * 16,
                kv_lens=[9216, 1, 2048, 2049, 512, 7777, 33, 9000, 4096, 1500, 300, 8191]
                + [6000, 2, 64, 4097],
                layer_ids=tuple(range(72)),
            ),
        ),
        Cell(
            why="ML4 32768 decode graph batch 1 (captured), 72 calls: one sequence at "
            "max_seq_len 32768, its newest position selected, in the 1024-column table's "
            "last column",
            spec=dict(
                _ML4_32K,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[32768],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 32768 decode graph batch 2 (captured), 72 calls: contexts of 1 and "
            "32768, swapped between the capture and the replay",
            spec=dict(
                _ML4_32K,
                graph=True,
                last_position=True,
                seq_rows=[1, 1],
                kv_lens=[1, 32768],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 32768 decode graph batch 4 (captured), 72 calls: contexts of 32737 "
            "(one position in the last column), 32736 (the last column unused), 2049, 33",
            spec=dict(
                _ML4_32K,
                graph=True,
                last_position=True,
                seq_rows=[1] * 4,
                kv_lens=[32737, 33, 2049, 32736],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 32768 decode graph batch 8 (captured), 72 calls: contexts across "
            "1..32768, most longer than top-k",
            spec=dict(
                _ML4_32K,
                graph=True,
                last_position=True,
                seq_rows=[1] * 8,
                kv_lens=[32768, 16384, 9217, 4097, 2047, 64, 24000, 30001],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 32768 decode graph batch 16 (captured), 72 calls: 13 sequences up to "
            "32768 and 3 padding rows (1-token dummies), pages at the far end of the pool",
            spec=dict(
                _ML4_32K,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[32768, 1, 2048, 2049, 512, 31111, 33, 9216, 16385, 1500, 300, 32767]
                + [20000]
                + [1] * 3,
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 32768 prefill at max_num_tokens: one 32768-token request, causal "
            "windows, the last pool layer; every row count 1..32768 is a bitwise prefix of it "
            "(about 8 s)",
            spec=dict(
                _ML4_32K,
                last_position=True,
                seq_rows=[32768],
                kv_lens=[32768],
                layer_ids=(71,),
                prefix_sweep=True,
            ),
        ),
        Cell(
            why="ML4 32768 context step at max_batch_size: 16 requests packed into 32768 "
            "rows, the KV sharers' pool row 35, pages at the far end of the pool",
            spec=dict(
                _ML4_32K,
                far_end=True,
                seq_rows=_ML4_32K_PACKED,
                kv_lens=_ML4_32K_PACKED,
                layer_ids=(35,),
            ),
        ),
        Cell(
            why="ML4 32768 mixed step: 3 context requests, then 13 generation rows with "
            "contexts up to 32768, in one 32768-row call",
            spec=dict(
                _ML4_32K,
                last_position=True,
                seq_rows=[16000, 10000, 6755] + [1] * 13,
                kv_lens=[16000, 10000, 6755]
                + [32768, 2, 2048, 2049, 4500, 32767, 1, 700, 16385, 30000, 9216, 129, 32737],
                layer_ids=(0,),
            ),
        ),
        Cell(
            why="ML4 32768 every layer id 0..71 at stride 72 x 32 = 2304, eager, 16 decode "
            "rows with contexts up to 32768",
            spec=dict(
                _ML4_32K,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[32768, 1, 2048, 2049, 512, 31111, 33, 9216, 16385, 1500, 300, 32767]
                + [20000, 2, 64, 32737],
                layer_ids=tuple(range(72)),
            ),
        ),
        Cell(
            why="ML4 1M decode graph batch 1 (captured), 72 calls: one sequence at max_seq_len "
            "1048576, its newest position 1048575 selected, in the 32768-column table's last "
            "column",
            spec=dict(
                _ML4_1M,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M decode graph batch 2 (captured), 72 calls: contexts of 1 and 1048576, "
            "swapped between the capture and the replay",
            spec=dict(
                _ML4_1M,
                graph=True,
                last_position=True,
                seq_rows=[1, 1],
                kv_lens=[1, 1048576],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M decode graph batch 4 (captured), 72 calls: contexts of 1048545 (one "
            "position in the last column), 1048544 (the last column unused), 2049, 33",
            spec=dict(
                _ML4_1M,
                graph=True,
                last_position=True,
                seq_rows=[1] * 4,
                kv_lens=[1048545, 33, 2049, 1048544],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M decode graph batch 8 (captured), 72 calls: contexts across 1..1048576, "
            "most longer than top-k",
            spec=dict(
                _ML4_1M,
                graph=True,
                last_position=True,
                seq_rows=[1] * 8,
                kv_lens=[1048576, 524288, 131073, 32769, 2047, 64, 700001, 1000000],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M decode graph batch 16 (captured), 72 calls: 13 sequences up to 1048576 "
            "and 3 padding rows (1-token dummies), pages at the far end of the pool",
            spec=dict(
                _ML4_1M,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000001, 33, 9216, 32769, 1500, 300]
                + [1048575, 524288, 1, 1, 1],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M prefill: the last 32768-token chunk of a 1048576-token prompt, behind "
            "its 1015808-token cached prefix, the last pool layer; every row count 1..32768 is "
            "a bitwise prefix of it",
            spec=dict(
                _ML4_1M,
                last_position=True,
                seq_rows=[32768],
                kv_lens=[1048576],
                layer_ids=(71,),
                prefix_sweep=True,
            ),
        ),
        Cell(
            why="ML4 1M context step at max_batch_size: 16 chunk tails packed into 32768 rows, "
            "cached prefixes up to 1048575, the KV sharers' pool row 35, pages at the far end "
            "of the pool",
            spec=dict(
                _ML4_1M,
                far_end=True,
                seq_rows=_ML4_1M_TAIL_ROWS,
                kv_lens=[1048576, 7, 32, 63, 65, 133, 2302, 2305, 2562, 33790, 66560, 133119]
                + [264192, 508191, 8193, 9011],
                layer_ids=(35,),
            ),
        ),
        Cell(
            why="ML4 1M mixed step: the last 16000 rows of a 1M prompt, 10000 behind a 490000 "
            "prefix and a fresh 6755, then 13 generation rows with contexts up to 1048576, in "
            "one 32768-row call",
            spec=dict(
                _ML4_1M,
                last_position=True,
                seq_rows=[16000, 10000, 6755] + [1] * 13,
                kv_lens=[1048576, 500000, 6755, 1048576, 2, 2048, 2049, 4500, 1048575, 1, 700]
                + [32769, 1000000, 131072, 129, 1048545],
                layer_ids=(0,),
            ),
        ),
        Cell(
            why="ML4 1M every layer id 0..71 at stride 72 x 32 = 2304, eager, 16 decode rows "
            "with contexts up to 1048576",
            spec=dict(
                _ML4_1M,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000001, 33, 9216, 32769, 1500, 300]
                + [1048575, 524288, 2, 64, 1048545],
                layer_ids=tuple(range(72)),
            ),
        ),
        Cell(
            why="ML4 1M 32772-column V2 table, graph batch 1 (captured), 72 calls: the boot "
            "capture's longest row, 1048576, its newest position in column 32767, four columns "
            "short of the table's end",
            spec=dict(
                _ML4_1M_V2,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M 32772-column V2 table, graph batch 16 (captured), 72 calls: contexts of "
            "1048576, 1048545 and 524288 beside short ones and 8 one-token rows, swapped "
            "end for end by the replay, pages at the far end of the pool",
            spec=dict(
                _ML4_1M_V2,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2049, 1048545, 33, 2048, 524288, 9216] + [1] * 8,
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M 32772-column V2 table, context step at max_batch_size: the 16 chunk "
            "tails of 32768 rows, cached prefixes up to 1048575, pool row 35, pages at the far "
            "end of the pool",
            spec=dict(
                _ML4_1M_V2,
                far_end=True,
                last_position=True,
                seq_rows=_ML4_1M_TAIL_ROWS,
                kv_lens=[1048576, 7, 32, 63, 65, 133, 2302, 2305, 2562, 33790, 66560, 133119]
                + [264192, 508191, 8193, 9011],
                layer_ids=(35,),
            ),
        ),
        Cell(
            why="ML4 1M 32772-column V2 table, mixed step: 3 context chunks behind cached "
            "prefixes up to 1032576, then 13 generation rows with contexts up to 1048576, "
            "in one 32768-row call, pool row 70",
            spec=dict(
                _ML4_1M_V2,
                last_position=True,
                seq_rows=[16000, 10000, 6755] + [1] * 13,
                kv_lens=[1048576, 500000, 6755, 1048576, 2, 2048, 2049, 4500, 1048575, 1, 700]
                + [32769, 1000000, 131072, 129, 1048545],
                layer_ids=(70,),
            ),
        ),
        Cell(
            why="ML4 1M tp4 decode graph batch 1 (captured), all 72 layer ids 0..71: one "
            "sequence at 1048576, its newest position 1048575 in the 32768-column table's last "
            "column",
            spec=dict(
                _ML4_1M,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids="all",
            ),
        ),
        Cell(
            why="ML4 1M tp4 decode graph batch 16 (captured), 72 calls: 13 sequences, 1048576 "
            "halved to 65536, then 32769, 32768, 16385, 9217, 4097, 2049, 2048 and 257, and 3 "
            "padding rows (1-token dummies), all 66609 pages the pool's top ones",
            spec=dict(
                _ML4_1M,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 524288, 262144, 131072, 65536, 32769, 32768, 16385, 9217]
                + [4097, 2049, 2048, 257, 1, 1, 1],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tp4 prefill chunk order: the last 32768-token chunk of a 1048576-token "
            "prompt, each row's selections split between the chunk's own rows up to it and the "
            "1015808-token cached prefix, pool rows 0, 35 and 70",
            spec=dict(
                _ML4_1M,
                order="chunk",
                last_position=True,
                seq_rows=[32768],
                kv_lens=[1048576],
                layer_ids=(0, 35, 70),
            ),
        ),
        Cell(
            why="ML4 1M tp4 packed chunk step: 16384, 8192, 4096 and 4096 rows behind cached "
            "prefixes of 1032192, 524288, 0 and 2048 in one 32768-row call, pool row 70, all "
            "49728 pages the pool's top ones",
            spec=dict(
                _ML4_1M,
                far_end=True,
                last_position=True,
                seq_rows=[16384, 8192, 4096, 4096],
                kv_lens=[1048576, 532480, 4096, 6144],
                layer_ids=(70,),
            ),
        ),
        *(
            Cell(
                why=f"dep4 1M boot capture, decode graph batch {b}, 72 calls: the warm-up's "
                f"longest dummy, 1048575 keys, beside {b - 1} one-key padding rows; replayed "
                "with one live row at 1048576 behind the padding, then the capture reversed",
                spec=dict(
                    _ML4_1M_DEP4,
                    graph=True,
                    last_position=True,
                    seq_rows=[1] * b,
                    kv_lens=[1048575] + [1] * (b - 1),
                    layer_ids=_ML4_STEP,
                    replay_kv_lens=([1048576] + [1] * (b - 1), [1] * (b - 1) + [1048575]),
                ),
            )
            for b in (1, 2, 4, 8, 16)
        ),
        Cell(
            why="dep4 1M decode graph batch 1 (captured), 72 calls: one sequence at max_seq_len "
            "1048576, its newest position 1048575 selected, in column 32767 of the 32772-column "
            "table",
            spec=dict(
                _ML4_1M_DEP4,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="dep4 1M decode graph batch 2 (captured), 72 calls: contexts of 1 and 1048576, "
            "swapped between the capture and the replay",
            spec=dict(
                _ML4_1M_DEP4,
                graph=True,
                last_position=True,
                seq_rows=[1, 1],
                kv_lens=[1, 1048576],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="dep4 1M decode graph batch 4 (captured), 72 calls: contexts of 1048545 (one "
            "position in column 32767), 1048544 (column 32767 unused), 2049, 33",
            spec=dict(
                _ML4_1M_DEP4,
                graph=True,
                last_position=True,
                seq_rows=[1] * 4,
                kv_lens=[1048545, 33, 2049, 1048544],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="dep4 1M decode graph batch 8 (captured), 72 calls: contexts across 1..1048576, "
            "past each power of two from 32768, most longer than top-k",
            spec=dict(
                _ML4_1M_DEP4,
                graph=True,
                last_position=True,
                seq_rows=[1] * 8,
                kv_lens=[1048576, 524288, 262145, 131073, 32769, 2047, 64, 1000001],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="dep4 1M decode graph batch 16 (captured), 72 calls: 13 sequences up to 1048576 "
            "and 3 padding rows (1-token dummies); their 122596 pages outnumber the pool and run "
            "through all of it, its top block included",
            spec=dict(
                _ML4_1M_DEP4,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000000, 33, 32769, 524289, 1500, 300]
                + [1048575, 262144, 1, 1, 1],
                layer_ids=_ML4_STEP,
            ),
        ),
        Cell(
            why="dep4 1M prefill: the last 32768-token chunk of a 1048576-token prompt, behind "
            "its 1015808-token cached prefix, the last pool layer; every row count 1..32768 is "
            "a bitwise prefix of it",
            spec=dict(
                _ML4_1M_DEP4,
                last_position=True,
                seq_rows=[32768],
                kv_lens=[1048576],
                layer_ids=(71,),
                prefix_sweep=True,
            ),
        ),
        Cell(
            why="dep4 1M context step at max_batch_size: 16 chunk tails of 16 rows, each behind "
            "a 65520-token cached prefix, the KV sharers' pool row 35, pages at the far end of "
            "the pool",
            spec=dict(
                _ML4_1M_DEP4,
                far_end=True,
                seq_rows=[16] * 16,
                kv_lens=[65536] * 16,
                layer_ids=(35,),
            ),
        ),
        Cell(
            why="dep4 1M mixed step: chunks of 16000 rows behind 500000 cached, 10000 fresh and "
            "6755 behind 1000000 cached, then 13 generation rows with contexts up to 1048576, in "
            "one 32768-row call",
            spec=dict(
                _ML4_1M_DEP4,
                last_position=True,
                seq_rows=[16000, 10000, 6755] + [1] * 13,
                kv_lens=[516000, 10000, 1006755, 1048576, 2, 2048, 2049, 4500, 1048575, 1, 700]
                + [524289, 30000, 9216, 129, 1048545],
                layer_ids=(0,),
            ),
        ),
        Cell(
            why="dep4 1M every layer id 0..71 at stride 72 x 32 = 2304, eager, 16 decode rows "
            "with contexts up to 1048576",
            spec=dict(
                _ML4_1M_DEP4,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000000, 33, 32769, 524289, 1500, 300]
                + [1048575, 262144, 2, 64, 1048545],
                layer_ids=tuple(range(72)),
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla decode graph batch 16 (captured), 72 calls over the 63-owner "
            "pool: stride 2016, each layer at its source's owner row, contexts up to 32768",
            spec=dict(
                _ML4_32K_DS,
                graph=True,
                far_end=True,
                seq_rows=[1] * 16,
                kv_lens=[32768, 1, 2048, 2049, 512, 31111, 33, 9216, 16385, 1500, 300, 32767]
                + [1] * 4,
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla mixed step over the 63-owner pool: 2 context requests and 6 "
            "generation rows, layer 35's owner row (the sharers' source) and the last",
            spec=dict(
                _ML4_32K_DS,
                seq_rows=[20000, 4000] + [1] * 6,
                kv_lens=[20000, 4000, 32768, 2, 2049, 9000, 30000, 777],
                layer_ids=(_ML4_DS_STEP[35], _ML4_DS_STEP[71], 62),
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla 64-token pages, decode graph batch 16 (captured), 72 calls over "
            "the 63-owner pool: stride 4032, each layer at its source's owner row, contexts up "
            "to 32768 in the 512-column table, pages at the far end of the 53958-block pool "
            "(last row 217558655, 10.1% of 2**31)",
            spec=dict(
                _ML4_32K_DS64,
                graph=True,
                far_end=True,
                seq_rows=[1] * 16,
                kv_lens=[32768, 1, 2048, 2049, 512, 31111, 33, 9216, 16385, 1500, 300, 32767]
                + [1] * 4,
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla 64-token pages, mixed step over the 63-owner pool: 2 context "
            "requests and 6 generation rows, layer 35's owner row (the sharers' source) and "
            "the last",
            spec=dict(
                _ML4_32K_DS64,
                seq_rows=[20000, 4000] + [1] * 6,
                kv_lens=[20000, 4000, 32768, 2, 2049, 9000, 30000, 777],
                layer_ids=(_ML4_DS_STEP[35], _ML4_DS_STEP[71], 62),
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla tep8/tp8/dep8 decode graph batch 16 (captured), 72 calls over the "
            "63-owner pool: 13 sequences up to 32768 and 3 padding rows (1-token dummies), "
            "pages at the far end of a 147456-block pool",
            spec=dict(
                _ML4_32K_DS_8GPU,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[32768, 1, 2048, 2049, 512, 31111, 33, 9216, 16385, 1500, 300, 32767]
                + [20000]
                + [1] * 3,
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla tep8/tp8/dep8 context step at max_batch_size: 16 requests packed "
            "into 32768 rows, layer 35's owner row (the sharers' source) and the last, pages at "
            "the far end of a 147456-block pool",
            spec=dict(
                _ML4_32K_DS_8GPU,
                far_end=True,
                seq_rows=_ML4_32K_PACKED,
                kv_lens=_ML4_32K_PACKED,
                layer_ids=(_ML4_DS_STEP[35], 62),
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla tep8/tp8/dep8 V2 table: decode graph batch 16 (captured), 72 "
            "calls, over the 1028-column block table the cache manager builds, pages at the far "
            "end of a 147456-block pool",
            spec=dict(
                _ML4_32K_DS_8GPU_V2,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[32768, 32767, 32737, 32736, 1, 2049, 16385, 9216, 2048, 33, 300, 31111]
                + [20000]
                + [1] * 3,
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla dep8 context step at max_batch_size: 16 requests packed into "
            "32768 rows, the KV sharers' source owner row 35 (as layer 35 and as layer 71, which "
            "reads it) and the last owner row, pages at the far end of a 147456-block pool",
            spec=dict(
                _ML4_32K_DS_8GPU,
                far_end=True,
                seq_rows=_ML4_32K_PACKED,
                kv_lens=_ML4_32K_PACKED,
                layer_ids=(_ML4_DS_STEP[35], _ML4_DS_STEP[71], 62),
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla dep8 V2 table, attention-DP padding: decode graph batch 16 "
            "(captured), 72 calls, one live sequence at 32768 behind 15 padding dummies, as a "
            "rank with one request replays the largest rank's batch",
            spec=dict(
                _ML4_32K_DS_8GPU_V2,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[32768] + [1] * 15,
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 fp8_ds_mla dep8 V2 table: context step at max_batch_size, 16 requests "
            "packed into 32768 rows over the 1028-column table, the sharers' source owner row "
            "35, pages at the far end of a 147456-block pool",
            spec=dict(
                _ML4_32K_DS_8GPU_V2,
                far_end=True,
                seq_rows=_ML4_32K_PACKED,
                kv_lens=_ML4_32K_PACKED,
                layer_ids=(_ML4_DS_STEP[35],),
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode graph batch 1 (captured), 72 calls over the 63-owner "
            "pool at stride 2016, each layer at its source's owner row: one sequence at "
            "1048576, its newest position 1048575 in column 32767 of the 32772-column table",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode graph batch 2 (captured), 72 calls at stride 2016: "
            "contexts of 1 and 1048576, swapped between the capture and the replay",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1, 1],
                kv_lens=[1, 1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode graph batch 4 (captured), 72 calls at stride 2016: "
            "contexts of 1048576, 1048545 (one position in column 32767), 2049 and 33",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 4,
                kv_lens=[1048576, 1048545, 2049, 33],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode graph batch 8 (captured), 72 calls at stride 2016: "
            "contexts across 1..1048576, most longer than top-k",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 8,
                kv_lens=[1048576, 524288, 131073, 32769, 2047, 64, 700001, 1000000],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode graph batch 16 (captured), 72 calls at stride 2016: "
            "13 sequences, 1048576 halved to 65536, then 32769 down to 257, and 3 padding rows "
            "(1-token dummies), all 66609 pages the 107917-block pool's top ones",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 524288, 262144, 131072, 65536, 32769, 32768, 16385, 9217]
                + [4097, 2049, 2048, 257, 1, 1, 1],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla mixed step's generation rows, eager: 13 rows whose requests "
            "are table rows 3..15 (req_idx from num_contexts 3), contexts up to 1048576, owner "
            "rows 35 (the sharers' source) and 62",
            spec=dict(
                _ML4_1M_DS,
                last_position=True,
                first_seq=3,
                seq_rows=[1] * 13,
                kv_lens=[1048576, 2, 2048, 2049, 4500, 1048575, 1, 700, 32769, 1000000, 131072]
                + [129, 1048545],
                layer_ids=(35, 62),
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla every owner row 0..62 at stride 2016, eager, 16 decode rows "
            "with contexts up to 1048576",
            spec=dict(
                _ML4_1M_DS,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000001, 33, 9216, 32769, 1500, 300]
                + [1048575, 524288, 2, 64, 1048545],
                layer_ids="owner_rows_0_62",
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode path, chunk order: the last 32768-row chunk of a "
            "1048576-token prompt in one call, each row's selections split between its chunk "
            "and the 1015808-token cached prefix, owner rows 0, 35 and 62; every row count "
            "1..32768 is a bitwise prefix of it (about 8.5 s)",
            spec=dict(
                _ML4_1M_DS,
                order="chunk",
                last_position=True,
                seq_rows=[32768],
                kv_lens=[1048576],
                layer_ids=(0, 35, 62),
                prefix_sweep=True,
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode path, packed chunks: 16384, 8192, 4096 and 4096 rows "
            "behind cached prefixes of 1032192, 524288, 0 and 2048 in one 32768-row call, "
            "owner row 62, all 49728 pages the pool's top ones",
            spec=dict(
                _ML4_1M_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16384, 8192, 4096, 4096],
                kv_lens=[1048576, 532480, 4096, 6144],
                layer_ids=(62,),
            ),
        ),
        Cell(
            why="ML4 1M fp8_ds_mla decode path, a 16-row tail beside a chunk: a 1048576-token "
            "prompt's last 16 rows and 32752 rows behind a 524304-token prefix in one "
            "32768-row call, owner row 35, all 50176 pages the pool's top ones",
            spec=dict(
                _ML4_1M_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16, 32752],
                kv_lens=[1048576, 557056],
                layer_ids=(35,),
            ),
        ),
        # The decode path's steps of a 1M prompt before its last chunk: 32768 rows behind a
        # cached prefix of 524288 (the first step past 2**19 context KV rows) to 983040; the
        # chunk behind 1015808 is the chunk-order cell above.
        *(
            Cell(
                why=f"ML4 1M fp8_ds_mla chunk behind {n - 32768} cached, decode path: a 1M "
                f"prompt's 32768-row chunk in one call over {n} KV rows, each row's selections "
                "split between its chunk and the cached prefix, owner rows 0, 35 and 62",
                spec=dict(
                    _ML4_1M_DS,
                    order="chunk",
                    last_position=True,
                    seq_rows=[32768],
                    kv_lens=[n],
                    layer_ids=(0, 35, 62),
                ),
            )
            for n in range(557056, 1015809, 32768)
        ),
        Cell(
            why="ML4 1M fp8_ds_mla chunk tail beside a fresh prompt, decode path: a 1040000-token "
            "prompt's last 24192 rows behind its 1015808-token cached prefix and a fresh 7-row "
            "prompt in one 24199-row call, in chunk order, owner rows 0, 35 and 62",
            spec=dict(
                _ML4_1M_DS,
                order="chunk",
                last_position=True,
                seq_rows=[24192, 7],
                kv_lens=[1040000, 7],
                layer_ids=(0, 35, 62),
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode graph batch 1 (captured), 72 calls over the "
            "63-owner pool at stride 2016, each layer at its source's owner row: one sequence "
            "at 1048576, its newest position 1048575 in column 32767 of the 32772-column table",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode graph batch 2 (captured), 72 calls at stride "
            "2016: contexts of 1 and 1048576, swapped between the capture and the replay",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1, 1],
                kv_lens=[1, 1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode graph batch 4 (captured), 72 calls at stride "
            "2016: contexts of 1048576, 1048545 (one position in column 32767), 2049 and 33",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 4,
                kv_lens=[1048576, 1048545, 2049, 33],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode graph batch 8 (captured), 72 calls at stride "
            "2016: contexts across 1..1048576, most longer than top-k",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 8,
                kv_lens=[1048576, 524288, 131073, 32769, 2047, 64, 700001, 1000000],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode graph batch 16 (captured), 72 calls at stride "
            "2016: 13 sequences, 1048576 halved down to 65536, then 32769 down to 257, and 3 "
            "padding rows (1-token dummies), all 66609 pages the 107917-block pool's top ones",
            spec=dict(
                _ML4_1M_DS,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 524288, 262144, 131072, 65536, 32769, 32768, 16385, 9217]
                + [4097, 2049, 2048, 257, 1, 1, 1],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla mixed step's generation rows, eager: 13 rows whose "
            "requests are table rows 3..15 (req_idx from num_contexts 3), contexts up to "
            "1048576, owner rows 35 (the sharers' source) and 62",
            spec=dict(
                _ML4_1M_DS,
                last_position=True,
                first_seq=3,
                seq_rows=[1] * 13,
                kv_lens=[1048576, 2, 2048, 2049, 4500, 1048575, 1, 700, 32769, 1000000, 131072]
                + [129, 1048545],
                layer_ids=(35, 62),
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla every owner row 0..62 at stride 2016, eager, 16 decode "
            "rows with contexts up to 1048576",
            spec=dict(
                _ML4_1M_DS,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000001, 33, 9216, 32769, 1500, 300]
                + [1048575, 524288, 2, 64, 1048545],
                layer_ids="owner_rows_0_62",
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode path, chunk order: the last 32768-row chunk of a "
            "1048576-token prompt in one call, each row's selections split between its chunk "
            "and the 1015808-token cached prefix, owner rows 0, 35 and 62; every row count "
            "1..32768 is a bitwise prefix of it (about 8.5 s)",
            spec=dict(
                _ML4_1M_DS,
                order="chunk",
                last_position=True,
                seq_rows=[32768],
                kv_lens=[1048576],
                layer_ids=(0, 35, 62),
                prefix_sweep=True,
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode path, packed chunks: 16384, 8192, 4096 and 4096 "
            "rows behind cached prefixes of 1032192, 524288, 0 and 2048 in one 32768-row call, "
            "owner row 62, all 49728 pages the pool's top ones",
            spec=dict(
                _ML4_1M_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16384, 8192, 4096, 4096],
                kv_lens=[1048576, 532480, 4096, 6144],
                layer_ids=(62,),
            ),
        ),
        Cell(
            why="ML4 1M tep4 fp8_ds_mla decode path, a 16-row tail beside a chunk: a "
            "1048576-token prompt's last 16 rows and 32752 rows behind a 524304-token prefix "
            "in one 32768-row call, owner row 35, all 50176 pages the pool's top ones",
            spec=dict(
                _ML4_1M_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16, 32752],
                kv_lens=[1048576, 557056],
                layer_ids=(35,),
            ),
        ),
        *(
            Cell(
                why=f"dep4 1M fp8_ds_mla boot graph {b} (captured), 72 calls at stride 2016, each "
                "layer at its source's owner row: the warm-up's longest dummy, 1048575 keys, "
                f"beside {b - 1} one-key fillers; replayed under attention DP with one live row at "
                f"1048576 behind {b - 1} rows of the one padding dummy, then on an idle rank (its "
                "attention-DP dummy behind the same padding)",
                spec=dict(
                    _ML4_1M_DEP4_DS,
                    graph=True,
                    last_position=True,
                    boot_capture=True,
                    seq_rows=[1] * b,
                    kv_lens=[1048575] + [1] * (b - 1),
                    layer_ids=_ML4_DS_STEP,
                    replay_kv_lens=([1048576] + [1] * (b - 1), [1] * b),
                    replay_pad=(b - 1, b - 1),
                ),
            )
            for b in (1, 2, 4, 8, 16)
        ),
        Cell(
            why="dep4 1M fp8_ds_mla decode graph 1 (captured), 72 calls at stride 2016, each layer "
            "at its source's owner row: one sequence at 1048576, its newest position 1048575 in "
            "column 32767 of the 32772-column table",
            spec=dict(
                _ML4_1M_DEP4_DS,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla decode graph 2 (captured), 72 calls at stride 2016: contexts "
            "of 1 and 1048576, swapped between the capture and the replay",
            spec=dict(
                _ML4_1M_DEP4_DS,
                graph=True,
                last_position=True,
                seq_rows=[1, 1],
                kv_lens=[1, 1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla decode graph 4 (captured), 72 calls at stride 2016: contexts "
            "of 1048545 (one position in column 32767), 33, 2049 and 1048544 (column 32767 "
            "unused); replayed reversed, then with 3 live rows behind one row of the padding dummy",
            spec=dict(
                _ML4_1M_DEP4_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 4,
                kv_lens=[1048545, 33, 2049, 1048544],
                layer_ids=_ML4_DS_STEP,
                replay_kv_lens=([1048544, 2049, 33, 1048545], [1048545, 2049, 33, 1]),
                replay_pad=(0, 1),
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla decode graph 8 (captured), 72 calls at stride 2016: contexts "
            "across 1..1048576, past each power of two from 32768; replayed reversed, then with 5 "
            "live rows behind 3 rows of the padding dummy",
            spec=dict(
                _ML4_1M_DEP4_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 8,
                kv_lens=[1048576, 524288, 262145, 131073, 32769, 2047, 64, 1000001],
                layer_ids=_ML4_DS_STEP,
                replay_kv_lens=(
                    [1000001, 64, 2047, 32769, 131073, 262145, 524288, 1048576],
                    [1000001, 64, 2047, 32769, 1048576, 1, 1, 1],
                ),
                replay_pad=(0, 3),
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla decode graph 16 (captured), 72 calls at stride 2016: 13 "
            "sequences, 1048576 halved to 65536, then 32769 down to 257, behind 3 rows of the "
            "padding dummy, all 66609 pages the 107917-block pool's top ones; replayed on the 13 "
            "reversed, then with 9 live rows behind 7 rows of the padding dummy",
            spec=dict(
                _ML4_1M_DEP4_DS,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 524288, 262144, 131072, 65536, 32769, 32768, 16385, 9217]
                + [4097, 2049, 2048, 257, 1, 1, 1],
                pad=3,
                layer_ids=_ML4_DS_STEP,
                replay_kv_lens=(
                    [257, 2048, 2049, 4097, 9217, 16385, 32768, 32769, 65536, 131072, 262144]
                    + [524288, 1048576, 1, 1, 1],
                    [1048576, 65536, 2049, 257, 524288, 32769, 9217, 131072, 2048] + [1] * 7,
                ),
                replay_pad=(3, 7),
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla mixed step's generation rows, eager: 13 rows whose requests "
            "are table rows 3..15 (req_idx from num_contexts 3), contexts up to 1048576, owner "
            "rows 35 (the sharers' source) and 62",
            spec=dict(
                _ML4_1M_DEP4_DS,
                last_position=True,
                first_seq=3,
                seq_rows=[1] * 13,
                kv_lens=[1048576, 2, 2048, 2049, 4500, 1048575, 1, 700, 32769, 1000000, 131072]
                + [129, 1048545],
                layer_ids=(35, 62),
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla every owner row 0..62 at stride 2016, eager, 16 decode rows "
            "with contexts up to 1048576",
            spec=dict(
                _ML4_1M_DEP4_DS,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000001, 33, 9216, 32769, 1500, 300]
                + [1048575, 524288, 2, 64, 1048545],
                layer_ids="owner_rows_0_62",
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla ctx rows, chunk order: the last 32768-row chunk of a "
            "1048576-token prompt in one call, each row's selections split between its chunk and "
            "the 1015808-token cached prefix, owner rows 0, 35 and 62; every row count 1..32768 "
            "is a bitwise prefix of it",
            spec=dict(
                _ML4_1M_DEP4_DS,
                order="chunk",
                last_position=True,
                seq_rows=[32768],
                kv_lens=[1048576],
                layer_ids=(0, 35, 62),
                prefix_sweep=True,
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla ctx rows, packed chunks: 16384, 8192, 4096 and 4096 rows "
            "behind cached prefixes of 1032192, 524288, 0 and 2048 in one 32768-row call, owner "
            "row 62, all 49728 pages the pool's top ones",
            spec=dict(
                _ML4_1M_DEP4_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16384, 8192, 4096, 4096],
                kv_lens=[1048576, 532480, 4096, 6144],
                layer_ids=(62,),
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla ctx rows, 16 chunk tails of 16 rows, each behind a 65520-token "
            "cached prefix (1048576 context KV, past 2**19), owner row 35, all 32768 pages the "
            "pool's top ones",
            spec=dict(
                _ML4_1M_DEP4_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16] * 16,
                kv_lens=[65536] * 16,
                layer_ids=(35,),
            ),
        ),
        Cell(
            why="dep4 1M fp8_ds_mla ctx rows, a 16-row tail beside a chunk: a 1048576-token "
            "prompt's last 16 rows and 32752 rows behind a 524304-token prefix in one 32768-row "
            "call, owner row 35, all 50176 pages the pool's top ones",
            spec=dict(
                _ML4_1M_DEP4_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16, 32752],
                kv_lens=[1048576, 557056],
                layer_ids=(35,),
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode graph batch 1 (captured), 72 calls over the "
            "63-owner pool at stride 2016, each layer at its source's owner row: one sequence "
            "at 1048576, its newest position 1048575 in column 32767 of the 32772-column table",
            spec=dict(
                _ML4_1M_TEP8_DS,
                graph=True,
                last_position=True,
                seq_rows=[1],
                kv_lens=[1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode graph batch 2 (captured), 72 calls at stride "
            "2016: contexts of 1 and 1048576, swapped between the capture and the replay",
            spec=dict(
                _ML4_1M_TEP8_DS,
                graph=True,
                last_position=True,
                seq_rows=[1, 1],
                kv_lens=[1, 1048576],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode graph batch 4 (captured), 72 calls at stride "
            "2016: contexts of 1048576, 1048545 (one position in column 32767), 2049 and 33",
            spec=dict(
                _ML4_1M_TEP8_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 4,
                kv_lens=[1048576, 1048545, 2049, 33],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode graph batch 8 (captured), 72 calls at stride "
            "2016: contexts across 1..1048576, most longer than top-k",
            spec=dict(
                _ML4_1M_TEP8_DS,
                graph=True,
                last_position=True,
                seq_rows=[1] * 8,
                kv_lens=[1048576, 524288, 131073, 32769, 2047, 64, 700001, 1000000],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode graph batch 16 (captured), 72 calls at stride "
            "2016: 13 sequences, 1048576 halved down to 65536, then 32769 down to 257, and 3 "
            "padding rows (1-token dummies), all 66609 pages the 147456-block pool's top ones",
            spec=dict(
                _ML4_1M_TEP8_DS,
                graph=True,
                far_end=True,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 524288, 262144, 131072, 65536, 32769, 32768, 16385, 9217]
                + [4097, 2049, 2048, 257, 1, 1, 1],
                layer_ids=_ML4_DS_STEP,
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla mixed step's generation rows, eager: 13 rows whose "
            "requests are table rows 3..15 (req_idx from num_contexts 3), contexts up to "
            "1048576, owner rows 35 (the sharers' source) and 62",
            spec=dict(
                _ML4_1M_TEP8_DS,
                last_position=True,
                first_seq=3,
                seq_rows=[1] * 13,
                kv_lens=[1048576, 2, 2048, 2049, 4500, 1048575, 1, 700, 32769, 1000000, 131072]
                + [129, 1048545],
                layer_ids=(35, 62),
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla every owner row 0..62 at stride 2016, eager, 16 decode "
            "rows with contexts up to 1048576, pages cycled through a 147456-block pool",
            spec=dict(
                _ML4_1M_TEP8_DS,
                last_position=True,
                seq_rows=[1] * 16,
                kv_lens=[1048576, 1, 2048, 2049, 512, 1000001, 33, 9216, 32769, 1500, 300]
                + [1048575, 524288, 2, 64, 1048545],
                layer_ids="owner_rows_0_62",
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode path, chunk order: the last 32768-row chunk of a "
            "1048576-token prompt in one call, each row's selections split between its chunk "
            "and the 1015808-token cached prefix, owner rows 0, 35 and 62; every row count "
            "1..32768 is a bitwise prefix of it (about 8.5 s)",
            spec=dict(
                _ML4_1M_TEP8_DS,
                order="chunk",
                last_position=True,
                seq_rows=[32768],
                kv_lens=[1048576],
                layer_ids=(0, 35, 62),
                prefix_sweep=True,
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode path, packed chunks: 16384, 8192, 4096 and 4096 "
            "rows behind cached prefixes of 1032192, 524288, 0 and 2048 in one 32768-row call, "
            "owner row 62, all 49728 pages the 147456-block pool's top ones",
            spec=dict(
                _ML4_1M_TEP8_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16384, 8192, 4096, 4096],
                kv_lens=[1048576, 532480, 4096, 6144],
                layer_ids=(62,),
            ),
        ),
        Cell(
            why="ML4 1M tep8 fp8_ds_mla decode path, a 16-row tail beside a chunk: a "
            "1048576-token prompt's last 16 rows and 32752 rows behind a 524304-token prefix "
            "in one 32768-row call, owner row 35, all 50176 pages the 147456-block pool's top "
            "ones",
            spec=dict(
                _ML4_1M_TEP8_DS,
                far_end=True,
                last_position=True,
                seq_rows=[16, 32752],
                kv_lens=[1048576, 557056],
                layer_ids=(35,),
            ),
        ),
    )

    note = """
    Metadata consumed: none. The op is a pure function of its arguments.

    The kernel launches one block row per `req_id` entry and writes
    `num_topk_tokens` columns at the output's row stride, which is
    `token_indices.size(1)`. Both counts are taken on trust; see `is_valid`.
    Rows are independent: an m-row call is bitwise the leading m rows of a
    longer one, driven at every m in 1..9216 and three times at every m in
    1..32768: the second time on the last chunk of a 1048576-token prompt, the
    third on that chunk in chunk order at stride 2016.

    Preconditions, each refused by `is_valid` because the op accepts the input
    and answers wrongly without raising (measured, except where marked read):
    - `num_topk_tokens == token_indices.size(1)`: narrower leaves each row's
      trailing columns unwritten; wider spills into the next row and past the
      output (read from the kernel, not driven).
    - `req_id.size(0) == token_indices.size(0)`: shorter leaves the trailing
      output rows unwritten; longer reads and writes past them (read from the
      kernel, not driven).
    - every `req_id` value is a row of `block_table`: the kernel reads
      `block_table[req_id[i]]` unchecked, so a request index past a sliced table
      reads the parent tensor's next row, and -1 the row before the slice.
    - `0 <= layer_id < stride_factor / block_size`: outside the page the row
      lands in a neighbouring page (layer 72 at stride 2304 is the next page's
      layer 0; -1 is the previous page's last layer).
    - every pool row fits int32 (Hard limits).
    The value checks read `req_id` and `block_table` on the host, which a CUDA
    graph capture forbids; a captured call is checked for shapes only, so check
    the data a replay reads eagerly.

    Mistral Large 4 passes `stride_factor = 72 * 32 = 2304` (a pool that
    interleaves all 72 layers' pages) and, as `layer_id`, the pool row of the
    layer whose latent it reads: a KV-sharing layer (39, 43, ..., 71) passes 35.
    Its FP8 target serves 9216 tokens (a 288-column block table); its NVFP4
    targets serve max_seq_len = max_num_tokens = 32768 with no chunked prefill
    (a 1024-column table, up to 32768 rows per call); on the fp8_ds_mla pool
    (`MistralLarge4CacheManager`) the stride is 63 x 32 = 2016 and `layer_id` the
    source's owner row, and a tep8, tp8 or dep8 rank's pool holds up to 147456
    pages. The cache manager pads the block table to 1028 columns at 32768,
    which only the "V2 table" cells drive; the columns past 1023 are never read
    below max_seq_len. Under attention DP (dep8) every rank replays the largest
    rank's graph batch, its own requests followed by 1-token padding dummies
    (`CUDAGraphRunner._get_padded_batch`, `_torch/pyexecutor/cuda_graph_runner.py`;
    read, not driven: the engine repeats one dummy request, the cells give each
    dummy a page of its own); rows are independent, and one live row behind 15
    dummies is a cell.
    Its NVFP4 tep4 target also serves max_seq_len 1048576 with chunked
    prefill: steps of up to 32768 rows,
    chunks behind cached prefixes, positions to 1048575, and the engine's
    32772-column table (cells at 32768 and 32772 columns).
    Its NVFP4 tp4 target makes the same calls at the same envelope. In index
    mode a chunk step's context rows reach the op after the q-split's
    all-gather has filled every row of the selection buffer, so one call covers
    all of the step's context rows. When the indexer is skipped (dense causal),
    the context rows read the block table instead, and only the generation
    rows behind them reach the op. The tp4 cells add every layer id 0..71 in
    one capture, a far-end batch-16 graph whose pages fit the pool, a 32768-row
    chunk whose rows select from both their own chunk and the cached prefix,
    and four chunks packed behind prefixes up to 1032192. Its NVFP4 dep4
    target serves max_seq_len 1048576 with chunked prefill too: steps of up to
    32768 rows per rank, chunks behind cached prefixes, positions to 1048575,
    and the DSA metadata's 16 x 32772 table (`block_table`, `max_num_sequences`
    rows by the V2 manager's `max_blocks_per_seq`). The metadata writes each
    step's rows only up to the step's widest request, -1 past a request's own
    pages, and the columns past that keep earlier steps' values (read in the
    DSA metadata's `prepare_for_indexer_k_cache`, not driven), which the dep4
    cells model as stale page ids. The boot captures every decode graph on a
    dummy of 1048575 keys beside 1-key padding rows
    (`_capture_generation_cuda_graphs` with `add_longest_dummy_request`); the
    dep4 cells capture those shapes and replay them on live data.

    On the fp8_ds_mla latent pool (`MistralLarge4CacheManager`: pages for the
    63 KV owners only) the tp4, tep4 and tep8 targets pass `stride_factor =
    len(kv_owner_layers) * tokens_per_block = 63 * 32 = 2016` and, as
    `layer_id`, `latent_pool_row(kv_source)`: 0..38 for layers 0..38, 35 for
    each sharer 39, 43, ..., 71, and 39..62 for the owners between (cells:
    `_ML4_DS_STEP`). Its block table is `md.block_table`, 16 rows by the
    engine's 32772 columns at max_seq_len 1048576 (1028 at 32768); positions
    reach 1048575, column 32767; tep8's page ids reach 147455 (the eight-rank
    bound; its 1M boot held 114326 blocks a rank). The "ML4 1M fp8_ds_mla"
    (tp4), "ML4 1M tep4 fp8_ds_mla" and "ML4 1M tep8 fp8_ds_mla" cells certify
    three call shapes:
    - `DecodeTarget._attention`: the step's 72 calls, captured at the padded
      batches 1/2/4/8/16 (cells at 1, 2, 4, 8 and 16 rows, replayed on new
      data);
    - `PrefillTarget._attention`: a mixed step's 1..16 generation rows, eager,
      `req_idx[tc:]`, whose request indices start at `num_contexts`;
    - the context rows of a step whose context KV (cached plus new over its
      context requests) exceeds 2**19 rows, when `PrefillTarget._attn_ctx`
      attends them straight off the pool rather than through a bf16
      workspace: every context row of the step in one call, up to 32768 rows
      (in index mode after the q-split's all-gather has filled the selection
      buffer), each chunk row selecting across its request's cached prefix
      and its own chunk, several requests packed. Each step a 1M prompt
      chunked at 32768 takes on this path is a tp4 cell: its chunks behind
      cached prefixes of 524288, 557056, ..., 983040 and 1015808, and a
      1040000-token prompt's last 24192 rows behind 1015808 packed with a
      fresh 7-row prompt. Context rows that address a bf16 workspace add
      their request's workspace offset instead and do not call the op.
    The dep4 target passes the same stride and owner rows over its own
    `md.block_table` (16 x 32772); the "dep4 1M fp8_ds_mla" cells certify its
    three call shapes:
    - `DecodeTarget._attention`: the step's 72 calls, captured at the padded
      batches 1/2/4/8/16 on the boot's warm-up batch (the 1048575-key longest
      dummy beside one-key fillers) and on live batches, and replayed on new
      data; eager (a peer rank prefilling) at any count 1..16, which the row
      independence above covers;
    - `PrefillTarget._attention`: a mixed step's 1..15 generation rows, eager,
      `req_idx[tc:]`, whose request indices start at `num_contexts`;
    - the context rows of a step whose context KV (cached plus new over its
      context requests) exceeds 2**19 rows, attended straight off the pool
      rather than through a bf16 workspace: every context row of the step in
      one call, up to 32768 rows, each chunk row selecting across its
      request's cached prefix and its own chunk, several requests packed.
      Context rows that address a bf16 workspace add their request's
      workspace offset instead and do not call the op.
    Under attention DP every rank replays the largest rank's graph batch: the
    rank's live rows first, then one CUDA-graph padding dummy (one key,
    position 0) repeated to the batch (`CUDAGraphRunner._get_padded_batch`
    appends the one request object; read, not driven in an engine), and an
    idle rank's attention-DP dummy (one key) ahead of that padding. Each
    padding row is a batch row of its own, so its request index is its own and
    its table row holds the dummy's one page: every padding row maps to the
    same pool rows. The cells replay that behind one, three, five, nine and
    thirteen live rows and on an idle rank, every row bitwise the reference.
    The table must be passed whole: a table sliced narrower than a request's
    pages maps every position past its last column to -1 without raising (read
    from the kernel, not driven), the same -1 the top-k padding carries.
    The binding calls `.contiguous()` on each input before the launch
    (`convertReqIndexToGlobal`, `cpp/tensorrt_llm/thop/convertReqIndexToGlobalOp.cpp`;
    read, not driven): only contiguous inputs were driven. The output is
    allocated per call: under CUDA-graph capture it comes from the graph's pool
    and each replay rewrites it.

    Hard limits (sm_103)
      pool row    block * stride_factor + layer_id * block_size + p % block_size,
                  int32. Boundary 2**31: at stride 2304, block 932067 from layer 40,
                  every layer from block 932068. Probed at block 932067: layer 39
                  is exact up to 2**31 - 1; layer 40 wraps to -2**31, and
                  `is_valid` refuses it. The FP8 envelope's 18432-block pool
                  reaches row 42467327, the NVFP4 target's 107917-block bound row
                  248640767 (11.6% of 2**31), the tep8, tp8 and dep8 targets'
                  147456-block bound on the 2016-stride fp8_ds_mla pool row 297271295
                  (13.8% of 2**31); the far-end cells hand requests each pool's top
                  pages: exact. Block 932067 of a 72-layer fp8 latent
                  pool sits past 1236949843968 bytes, 4.16 times a GB300's
                  297020948480. The tp4 boot pools hold 74183 slots at 32768
                  (2373856 tokens) and 73500 slots at 1048576 (2352000 tokens):
                  the top row is 74182 * 2304 + 71 * 32 + 31 = 170917631, 7.96%
                  of 2**31, which is inside the 107917-block cells.
                  At the 63 owners' stride 2016: block 1065220 from owner row 4,
                  every row from block 1065221. Probed at block 1065220: row 3 is
                  exact up to 2**31 - 1; row 4 wraps to -2**31, and `is_valid`
                  refuses it. The fp8_ds_mla tp4 boot at 1048576 held 69011
                  blocks, top row 69010 * 2016 + 62 * 32 + 31 = 139126175 (6.48% of
                  2**31), the tep4 boot 72757 blocks (2328224 tokens), top row
                  72756 * 2016 + 62 * 32 + 31 = 146678111 (6.83%), the dep4 boot
                  64633 blocks per rank, top row 64632 * 2016 + 62 * 32 + 31 =
                  130300127 (6.07%), the tep8 boot 114326 blocks (3658432
                  tokens), top row 114325 * 2016 + 62 * 32 + 31 = 230481215
                  (10.73%); the 107917-block cells reach 107916 * 2016 + 62 * 32
                  + 31 = 217560671 (10.13%) and the 147456-block cells 147455 *
                  2016 + 62 * 32 + 31 = 297271295 (13.84%), and their far-end
                  cells hand requests the pool's top pages: exact.
      fp32        no index passes through fp32. Every ML4 cell's output names pool
                  rows past 2**24 + 1 (from block 7282 at stride 2304, 8323 at
                  stride 2016), which its test asserts: exact.
      offsets     row * stride in int64 for `token_indices`, `block_table` and
                  the output: no 32-bit product (one would wrap at 2**31 / 2048 =
                  1048576 rows). 32768 x 2048 = 2**26 elements in the envelope;
                  the table offset req * 32772 + column is at most 524347 at 16
                  rows.
      position    p and p // block_size in int32, no product: boundary 2**31,
                  not reached, not probed. Positions 0..1048575 in the 1M
                  envelope, table columns 0..32767 of the engine's 32772; the
                  table's row stride enters in int64. Every position is below
                  2**24 and none passes through fp32.
      rows        one grid row per `req_id` entry: 2**31 - 1 rows; 32768 in the
                  envelope.

    Not in any cell at stride 2016: a pool past 147456 blocks (past 107917 at
    tp4, tep4 and dep4), and a captured call of the context-row shape (those
    calls run eager).

    The grouped variant (`convert_req_index_to_global_grouped`, one launch for
    several layers) is a different op and is not covered here.
    """

    def __call__(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.convert_req_index_to_global(
            req_id,
            block_table,
            token_indices,
            block_size,
            num_topk_tokens,
            stride_factor,
            layer_id,
        )

    def reference(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        """The kernel's integer map, in int64 so it cannot wrap.

        `num_topk_tokens` is not read: the reference covers every column of
        `token_indices`, which `is_valid` requires to be the same count.
        """
        tok = token_indices.long()
        max_blocks = block_table.shape[1]
        block = torch.div(tok, block_size, rounding_mode="floor")
        in_block = tok - block * block_size + layer_id * block_size
        valid = (tok >= 0) & (block < max_blocks)
        rows = req_id.long().unsqueeze(1).expand_as(tok)
        base = block_table.long()[rows, block.clamp(0, max_blocks - 1)]
        valid &= base >= 0
        out = torch.where(valid, base * stride_factor + in_block, torch.full_like(tok, -1))
        return out.to(torch.int32)

    def is_valid(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> None:
        # The output's row stride is `token_indices.size(1)`, but the kernel
        # walks `num_topk_tokens` columns. Wider spills each row's surplus into
        # the next row and past the end of the output; narrower leaves the
        # trailing columns uninitialized. Neither raises.
        assert num_topk_tokens == token_indices.shape[1], (
            f"num_topk_tokens must equal token_indices.size(1); got {num_topk_tokens} "
            f"against a {tuple(token_indices.shape)} operand"
        )
        # The row count is `req_id.size(0)`, not `token_indices.size(0)`. A
        # longer `req_id` reads past `token_indices` and writes past the output;
        # a shorter one leaves the extra output rows uninitialized.
        assert req_id.shape[0] == token_indices.shape[0], (
            f"req_id must hold exactly one entry per row of token_indices; got "
            f"{req_id.shape[0]} for {token_indices.shape[0]} rows"
        )
        # `layer_id * block_size` is added inside the page unchecked: a layer
        # outside it addresses a neighbouring page's rows.
        assert 0 <= layer_id and (layer_id + 1) * block_size <= stride_factor, (
            f"layer_id {layer_id} is outside a page of {stride_factor // block_size} layers "
            f"(stride_factor {stride_factor}, block_size {block_size})"
        )
        # The rest reads values off the device, which a graph capture forbids.
        if torch.cuda.is_current_stream_capturing() or not req_id.numel():
            return
        reads = [req_id.min(), req_id.max()]
        if block_table.numel():
            reads.append(block_table.max())
        lo, hi, *top_block = torch.stack(reads).tolist()
        # `block_table[req_id[i]]` is read unchecked: past a sliced table it
        # reads the parent tensor's rows.
        assert 0 <= lo and hi < block_table.shape[0], (
            f"req_id values {lo}..{hi} must be rows of block_table, which has "
            f"{block_table.shape[0]} rows"
        )
        # int32 arithmetic on the device: a pool row past 2**31 - 1 wraps silently.
        if top_block:
            top = top_block[0] * stride_factor + (layer_id + 1) * block_size
            assert top <= 2**31, (
                f"pool row {top - 1} does not fit int32; the kernel wraps it silently"
            )


convert_req_index_to_global = ConvertReqIndexToGlobal()
