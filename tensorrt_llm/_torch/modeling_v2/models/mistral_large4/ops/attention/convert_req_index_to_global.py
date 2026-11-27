# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a DSA request-local KV position to its row in the paged latent KV pool."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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

#: The tp4 and tep4 deployments on the fp8_ds_mla pool at 1048576: the engine's 32772-column table
#: and the 107917-block bound, at the 63 owners' stride 2016. Their boots at this envelope held
#: 69011 blocks (2208352 tokens; tp4) and 72757 blocks (2328224 tokens; tep4) per rank, inside
#: the bound.
_ML4_1M_DS = dict(_ML4_1M_V2, pool_layers=63)

#: The tep8 deployment on the fp8_ds_mla pool at 1048576: the engine's 32772-column table and the
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

#: The dep4 deployment on the fp8_ds_mla pool at 1048576: the same 16 x 32772 table and
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
    # The tp4 deployment ("ML4 1M tp4") makes the same calls at the same envelope:
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
    # The tp4 and tep4 deployments on the fp8_ds_mla pool ("ML4 1M fp8_ds_mla" and
    # "ML4 1M tep4 fp8_ds_mla") pass stride 63 x 32 = 2016 and each layer's
    # `latent_pool_row(kv_source)` (`_ML4_DS_STEP`) at three sites: the step's 72
    # calls from `DecodeTarget._attention`; a mixed step's generation rows from
    # `PrefillTarget._attention`, whose requests are the table's rows from
    # `num_contexts` on (`first_seq`); and, on a step whose context KV exceeds
    # 2**19 rows, every context row of the step in one call from
    # `PrefillTarget._attn_ctx`. `layer_ids="owner_rows_0_62"` is every owner row
    # of the 63-owner page, one call each.
    #
    # The NVFP4 dep4 deployment at max_seq_len 1048576 ("dep4 1M") keeps those call
    # sites, each rank running its own requests, and steps of at most 32768
    # rows; what grows is the position (to 1048575) and the table (the 32772
    # columns the engine allocates). A chunk's rows sit behind the request's
    # cached prefix, so a step's positions reach far past its own row count.
    # The engine captures each decode graph at boot on its longest dummy
    # request, max_seq_len - 1 = 1048575 keys, beside 1-key padding rows, and
    # replays it on the step's data: `replay_kv_lens` names the replays' contexts
    # (without it a graph cell replays the capture's contexts reversed).
    #
    # The dep4 deployment on the fp8_ds_mla pool ("dep4 1M fp8_ds_mla") passes stride
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
    # The tep8 deployment on the fp8_ds_mla pool passes stride 63 x 32 = 2016 and
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
    )

    note = """
    Maps request-local top-k positions to global paged KV pool rows via
    `torch.ops.trtllm.convert_req_index_to_global`. Guards int32 row overflow,
    column/req_id mismatches, and out-of-range req_id or layer_id.
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
