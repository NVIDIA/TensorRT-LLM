# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged FP8 MQA logits: the DSA indexer's decode scores, in one CuTe DSL call."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Mistral Large 4's indexer: 32 heads of 128; the K cache holds 32 tokens per page,
#: each as 128 fp8 bytes plus one fp32 scale (132 bytes). It scores
#: one token per sequence (no speculation) into rows of `max_seq_len` columns:
#: 9216 for the FP8 target, 32768 for the NVFP4 target (`_ML4_32K` below), and
#: 1048576 for the NVFP4 target at the checkpoint's own context (`_ML4_1M`). In
#: its cells `layout` says how the cache view is made: "v2" and
#: "v1" rebuild the per-layer views the two DSA cache managers hand out (see
#: `note`), "manager_v2" and "manager_v1" take the view and the block ids from
#: a real manager at `layer` (built for `max_num_tokens`, 9216 unless the cell
#: says, and for `max_seq_len`, `max_context_len` unless the cell says).
#: `pool_layers` is the number of indexer rows in
#: the pool, `pool_row` the row read, `num_blocks` the pool's slot count, and
#: `tail` puts the sequences' pages in the pool's last blocks. The block table
#: is `table_rows` rows of `table_width` entries, the manager's
#: `max_blocks_per_seq` (measured: 292 for V2, 288 for V1, at 9216 tokens;
#: 1028 for V2, 1024 for V1, at 32768; 32772 for V2 at 1048576), the
#: call's sequences start at row `first_seq`, and its query and weight rows
#: come after `q_offset` rows of the same buffers, as the generation rows of a
#: mixed step do. `capture_lens` captures the call in a CUDA graph at those
#: lengths and replays it at `context_lens`. `padding` is the number of
#: trailing rows that repeat one dummy sequence's table row and length, as a
#: padded graph batch repeats one dummy request (its single key is read off
#: `add_dummy_requests`' default of one token, not measured). `past` is the
#: byte offset into the pool the highest page read must lie beyond.
_ML4 = dict(heads=32, head_dim=128, tokens_per_block=32, next_n=1, max_context_len=9216)

#: The largest pools an FP8-target rank can hold, deliberate overestimates:
#: about 24.7 GiB is left per GB300 after the 252.84 GiB of weights, and both
#: DSA cache managers take 47280 bytes per token with 44 indexer rows and 50976
#: with 72 (measured), so at most about 17.5K and 16.3K blocks of 32 tokens.
_ML4_POOL_44 = dict(layout="v2", pool_layers=44, num_blocks=18432, table_width=292)
_ML4_FAR_44 = dict(pool_layers=44, num_blocks=18432, tail=True, past=2**31)
_ML4_FAR_72 = dict(pool_layers=72, num_blocks=16384, tail=True, past=2**32)

#: Every edge of a page (32 keys), the schedule block (64), a compute block
#: (128) and a two-block step (256), and the row's full width.
_ML4_EDGES = [1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 2049, 9215, 9216]

#: The NVFP4 target: rows of 32768 columns, 1024 pages per sequence.
_ML4_32K = dict(heads=32, head_dim=128, tokens_per_block=32, next_n=1, max_context_len=32768)

#: The NVFP4 target's pools, in the engine-default V2 layout, the table V2's
#: `max_blocks_per_seq` at 32768 tokens wide. A boot gate read 2373024 KV
#: tokens per rank, `max_num_blocks` x 32, and `max_num_blocks` is the pool's
#: slot count (measured: a V2 manager built for a 2373024-token quota held
#: 78051 slots and reported 78051), so that pool is 74157 blocks: the far
#: cell's. The large cell's 109568 is more than a GB300 can hold beside the
#: ~143 GiB of weights per tep4 rank (about 94660 blocks at 47280 bytes per
#: token).
#: Graph and edge cells take 32768 blocks, twice the 16384 pages that 16
#: sequences of 32768 keys read.
_ML4_32K_POOL_44 = dict(layout="v2", pool_layers=44, num_blocks=32768, table_width=1028)
_ML4_32K_FAR_44 = dict(layout="v2", pool_layers=44, table_width=1028, pool_row=43, tail=True)

#: The largest pool a rank of the eight-GPU targets can hold, a bound: a tep8 or
#: tp8 rank keeps 73.61 GiB of weights and a dep8 rank 89.29 GiB (its 16 routed
#: experts and the replicated rest), computed from the checkpoint's shard headers
#: under each target's split rules, so 203.01 and 187.33 GiB of a 276.62 GiB GB300
#: are left, 144075 and 132949 V2 slots of 32 tokens at 47280 bytes per token. V2
#: hands out about 5% more slots than its token quota, and 147456 clears both
#: readings. No boot has measured the pool at that bound.
_ML4_32K_8GPU_MAX_BLOCKS = 147456

#: The edges of the schedule block (64) and a two-block step (256), the
#: top-k's 2048, 4096, the FP8 envelope's 9216, half the row, the row's last
#: schedule block and its full width.
_ML4_32K_EDGES = [
    *(1, 64, 65, 255, 256, 257, 2048, 2049, 4097, 9216, 9217),
    *(16384, 16385, 32704, 32767, 32768),
]

#: The NVFP4 target at 64-token pages: 8448-byte pages (64 x 132), 512 pages per sequence,
#: and a V2 table of 516 columns (32768 / 64 rounded up past the manager's generation
#: headroom to a multiple of 4). Each pool holds its 32-token counterpart's tokens in half
#: the blocks: 16384 for the graph and mixed-step cells, 37078 for the measured 2373024-token
#: pool, 54784 for the large one.
_ML4_32K_P64 = dict(_ML4_32K, tokens_per_block=64)
_ML4_32K_P64_POOL_44 = dict(layout="v2", pool_layers=44, num_blocks=16384, table_width=516)
_ML4_32K_P64_FAR_44 = dict(layout="v2", pool_layers=44, table_width=516, pool_row=43, tail=True)

#: The NVFP4 target at its checkpoint's context (max_seq_len 1048576, chunked
#: prefill in 32768-token steps, max_batch_size 16, per rank for dep4): the
#: target passes `max_context_len = md.max_seq_len` (its `_max_ctx_len`,
#: frozen into every capture), so every call scores rows of 1048576 columns,
#: 32768 pages per sequence.
_ML4_1M = dict(heads=32, head_dim=128, tokens_per_block=32, next_n=1, max_context_len=1048576)

#: DSpark K=3 includes overlap/verification headroom at the 1M envelope.
#: Four causal rows per request, each padded to a 1048832-column stride.
_ML4_1M_DSPARK = dict(_ML4_1M, next_n=4, max_context_len=1048584)

#: Its pools, in the V2 layout, the table the V2 manager's `max_blocks_per_seq`
#: at 1048576 tokens wide (32772, measured on the manager: ceil((1048576 + 1) /
#: 32) padded to a multiple of 4). The tp4 boot gate read 2373856 KV tokens per
#: rank, 74183 blocks: room for two sequences of 1048576 keys beside shorter
#: ones, so the populations stay within it. `padded_to` puts the `padding`
#: rows of a 1-key dummy after `context_lens` (and `capture_lens`).
_ML4_1M_POOL_44 = dict(layout="v2", pool_layers=44, num_blocks=74183, table_width=32772)
_ML4_1M_FAR_44 = dict(layout="v2", pool_layers=44, table_width=32772, pool_row=43, tail=True)

#: tep4's pool, the delivered 74157 blocks in the V2 layout (13.8 GB of indexer
#: rows; two sequences of 1048576 keys take 65536 of its pages), read through
#: the table `KVCacheManagerV2` allocates at 1048576 tokens: ceil(1048577 / 32)
#: = 32769 pages, padded to a multiple of 4 (the manager cell measures it).
#: tep4's cells use it; it differs from `_ML4_1M_POOL_44` (tp4's boot pool of
#: 74183 blocks) only in `num_blocks`.
_ML4_1M_POOL_44_TEP4 = dict(layout="v2", pool_layers=44, num_blocks=74157, table_width=32772)

#: A dep4 rank's pool at 1048576, rounded up: 65536 blocks in the V2 layout
#: (12.2 GB of indexer rows). The dep4 boot gate at 1048576 read 1983872 KV
#: tokens per rank, `max_num_blocks` x 32, and V2's `max_num_blocks` is its
#: slot count, so that pool is 61996 slots (58557 under the accuracy gate's
#: settings); the far cell's 74157 blocks bound it too. The table is the one
#: `KVCacheManagerV2` allocates at 1048576 tokens: ceil(1048577 / 32) = 32769
#: pages, padded to a multiple of 4 (the manager cell measures it).
_ML4_1M_DEP4_POOL_44 = dict(layout="v2", pool_layers=44, num_blocks=65536, table_width=32772)

#: The tep8 deployment makes tep4's call at 1048576 (its 32 indexer heads
#: replicated on every rank) on a rank's own pool: the delivered tep8 boot read
#: 3680512 KV tokens per rank, 115016 V2 slots of 32 tokens (`max_num_blocks` is
#: the slot count, as for tep4's 74157), and `_ML4_32K_8GPU_MAX_BLOCKS` bounds
#: it. Same view and table as `_ML4_1M_POOL_44_TEP4`, the slot count per cell.
_ML4_1M_TEP8_BOOT_BLOCKS = 115016
_ML4_1M_TEP8_44 = dict(layout="v2", pool_layers=44, table_width=32772)


class CuteDslFp8PagedMqaLogits(OpWrapper):
    """Score each decode query token against its own sequence's paged keys.

    `q` is `[B, next_n, H, D]` fp8; `kv_fused` is the indexer K cache
    `[num_pages, page, 1, D + 4]` uint8, read through `block_table[b]`. Returns
    `[B * next_n, max_context_len]` fp32 with

        logits[b * next_n + t, j] = k_scale[j] * sum_h weights[b * next_n + t, h]
                                    * relu(q[b, t, h, :] . k[j, :])

    where `k` / `k_scale` are sequence `b`'s keys. Only the causal part of
    each row, `j < context_lens[b] - next_n + t + 1`, is defined; that is all
    the decode top-k reads. As with `fp8_mqa_logits`, the q-side scale must be
    folded into `weights`.

    The result is a view of a shared arena buffer, not a fresh tensor: the
    next call returns the same storage and overwrites it. Consume the logits
    before scoring again.

    Fusion boundary: the q.k products, relu, head weighting, head sum and key
    scale happen inside the call. The caller owns the q-side quantization and
    weight folding, the key cache writes, the schedule and the top-k after it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # `max_context_len` is the engine's max sequence length, so most of each
    # row is past the sequence's end. Context lengths are the sequences' KV
    # lengths after this step.
    #
    # Mistral Large 4 (max_batch_size 16, no speculation) scores 1..16 sequences
    # in CUDA-graph decode batches 1, 2, 4, 8 and 16 (padded), and the
    # generation rows of a mixed step eagerly. The FP8 target's cells (max_seq_len
    # 9216) capture every graph batch and replay it on new data, read the
    # engine-default V2 view of the largest 44-row pool, take the view and the
    # block ids from both real managers, and address the far end of the
    # largest pool in both layouts at 44 and 72 rows. The NVFP4 target's
    # ("NVFP4 32768", max_seq_len 32768) do the same for rows of 32768 keys in
    # its V2 layout: every graph batch replayed at lengths up to 32768, a mixed
    # step after 32765 prompt rows, the real V2 manager at 32768, and the far
    # end of its measured pool (past 2^33 bytes) and of a pool no tep4 rank
    # can hold (past 2^34, `large`). The "64-token pages" cells repeat those NVFP4
    # cells over 64-token pages: the same lengths, rows and replays, a 516-column
    # table and pools of the same bytes in half the blocks. The eight-GPU cell
    # ("NVFP4 32768 tep8/tp8/dep8") is the call of the tep8, tp8 and dep8
    # deployments, the 32 indexer heads whole on every rank (replicated at tep8 and
    # tp8, data parallel at dep8), on the larger pool a rank holds beside its
    # smaller share of the weights -- the far end of the 147456-block bound
    # (past 2^34, `large`). The NVFP4 target at max_seq_len 1048576 with
    # chunked prefill ("NVFP4 1M", tp4: the indexer is replicated, so every
    # rank scores all 32 heads) freezes 1048576 into the capture: every graph
    # batch replayed with one or two rows at 1M and the rest shorter, inside
    # its boot pool of 74183 blocks; the 16-row batch padded after 11 live
    # rows; a mixed step's 3 generation rows to 1M after 32765 prompt rows; the
    # far end of the boot pool; and, `large`, the real V2 manager at 1048576
    # built for the boot's 2373856-token quota (V2 keeps max_seq_len 1048576
    # only for a quota of at least 32769 blocks, about 50 GiB) and the far end
    # of the 109568-block pool.
    #
    # The NVFP4 tep4 deployment at max_seq_len 1048576 with chunked prefill
    # ("NVFP4 1M") keeps both call sites (DecodeTarget._indexer, captured;
    # PrefillTarget._indexer's generation rows of a mixed step, eager); what
    # grows is the row (1048576 columns), the context lengths (to 1048576) and
    # the table (32772 columns). Its cells replay every graph batch at lengths
    # up to 1048576 on the 44-row V2 view of the delivered 74157-block pool,
    # score a mixed step's generation rows after 32765 chunk rows, take the
    # view and the table from the real V2 manager built at 1048576 (`large`:
    # it took 115 GiB of the device), and read a 1048576-key sequence from the
    # pool's last pages. The kernel is compiled once for every row and table
    # width (both are symbolic in `CuteDSLPagedMQALogitsRunner._compile`), so
    # the 1M rows run the kernel the 32768 rows do. The tep8 deployment at 1048576
    # ("NVFP4 1M tep8") makes the tep4 call; its cells are the tep4 cells plus
    # a rank's own pools: boot's capture shape on the 115016-block pool the
    # tep8 boot measured, and a 1048576-key sequence at the far end of the
    # 147456-block bound (past 2^34), both `large` (pool and the test's copy
    # of it pass 40 GiB).
    #
    # The NVFP4 dep4 deployment at max_seq_len 1048576 with chunked prefill
    # ("dep4 1M") keeps both call sites (DecodeTarget._indexer, captured;
    # PrefillTarget._indexer's generation rows of a mixed step, eager); what
    # grows is the row (1048576 columns), the context lengths (to 1048576) and
    # the table (32772 columns). Boot reaches it first: the engine's CUDA-graph
    # warm-up captures each decode batch on its longest dummy request, 1048575
    # keys in row 0, beside 1-key fillers, and those shapes are replayed first
    # ("dep4 1M boot capture"). The other cells replay every graph batch at
    # lengths up to 1048576 on the 44-row V2 view of a 65536-block pool,
    # attention DP's one live row behind 1-key padding rows at batches 2, 4, 8
    # and 16, score a mixed step's generation rows after 32765 and 32753 chunk
    # rows, take the view and the table from the real V2 manager built at
    # 1048576 (`large`), and read a 1048576-key sequence from the last pages of
    # a 74157-block pool. The kernel is compiled once for every row and table
    # width (both, and the logits row stride, are symbolic in
    # `CuteDSLPagedMQALogitsRunner._compile`), so the 1M rows run the kernel
    # the 32768 rows do.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="DSpark 1M verify mixed: 60 rows after 32708 prompt rows",
            spec=dict(
                _ML4_1M_DSPARK,
                **_ML4_1M_POOL_44,
                pool_row=0,
                first_seq=1,
                table_rows=16,
                q_offset=32708,
                context_lens=[1048584, 2048, 2049, 32, 33] * 3,
            ),
        ),
        Cell(
            why="DSpark 1M verify DEP padding: one live request and 60 dummy rows",
            spec=dict(
                _ML4_1M_DSPARK,
                **_ML4_1M_POOL_44,
                pool_row=43,
                capture_lens=[1048583] + [4] * 15,
                context_lens=[1048575] + [4] * 15,
                padding=15,
            ),
        ),
        Cell(
            why="DSpark verify mixed: 15 sequences after 32708 prompt rows, padded table rows",
            spec=dict(
                _ML4_32K,
                next_n=4,
                max_context_len=32776,
                **_ML4_32K_POOL_44,
                pool_row=0,
                first_seq=1,
                table_rows=16,
                q_offset=32708,
                context_lens=[32776, 2048, 2049, 32, 33] * 3,
            ),
        ),
        Cell(
            why="DSpark verify dummy padding at one-key context, each query's causal end",
            spec=dict(
                _ML4_32K,
                next_n=4,
                max_context_len=32776,
                **_ML4_32K_POOL_44,
                pool_row=43,
                padding=7,
                capture_lens=[4] * 9 + [1] * 7,
                context_lens=[32776, 2049, 33, 65, 257, 4097, 9217, 16385, 32767] + [1] * 7,
            ),
        ),
        Cell(
            why="NVFP4 32768, decode graph batch 1: the row's full width, captured at 5000, replayed",
            spec=dict(
                _ML4_32K, **_ML4_32K_POOL_44, pool_row=43, capture_lens=[5000], context_lens=[32768]
            ),
        ),
        Cell(
            why="NVFP4 32768, decode graph batch 16: every schedule / block edge to 32768, replayed",
            spec=dict(
                _ML4_32K,
                **_ML4_32K_POOL_44,
                pool_row=31,
                capture_lens=[100 + 2000 * i for i in range(16)],
                context_lens=list(_ML4_32K_EDGES),
            ),
        ),
        Cell(
            why="NVFP4 32768, decode graph batch 16 padded: 11 sequences + 5 rows of a 1-key dummy",
            spec=dict(
                _ML4_32K,
                **_ML4_32K_POOL_44,
                pool_row=40,
                capture_lens=[1] * 16,
                context_lens=[32768, 32767, 30001, 24576, 16385, 9217, 8191, 2049, 500, 77, 2]
                + [1] * 5,
                padding=5,
            ),
        ),
        Cell(
            why="NVFP4 32768, mixed step, eager: 3 generation rows after 32765 prompt rows",
            spec=dict(
                _ML4_32K,
                **_ML4_32K_POOL_44,
                pool_row=12,
                table_rows=16,
                first_seq=2,
                q_offset=32765,
                context_lens=[2049, 777, 32768],
            ),
        ),
        Cell(
            why="NVFP4 32768, engine V2 manager (the default), 44 rows, layer 71: 3 sequences",
            spec=dict(
                _ML4_32K,
                layout="manager_v2",
                pool_layers=44,
                layer=71,
                max_tokens=65536,
                max_num_tokens=32768,
                table_width=1028,
                table_rows=16,
                context_lens=[32768, 9217, 1],
            ),
        ),
        Cell(
            why="64-token pages, NVFP4 32768, mixed step, eager: 3 generation rows after 32765 "
            "prompt rows",
            spec=dict(
                _ML4_32K_P64,
                **_ML4_32K_P64_POOL_44,
                pool_row=12,
                table_rows=16,
                first_seq=2,
                q_offset=32765,
                context_lens=[2049, 777, 32768],
            ),
        ),
        Cell(
            why="64-token pages, NVFP4 32768, engine V2 manager (the default), 44 rows, layer 71: "
            "3 sequences",
            spec=dict(
                _ML4_32K_P64,
                layout="manager_v2",
                pool_layers=44,
                layer=71,
                max_tokens=65536,
                max_num_tokens=32768,
                table_width=516,
                table_rows=16,
                context_lens=[32768, 9217, 1],
            ),
        ),
        Cell(
            why="dep4 1M, far pool V2 at 74157 blocks: a 1048576-key sequence's pages past 2^33",
            spec=dict(
                _ML4_1M,
                layout="v2",
                pool_layers=44,
                table_width=32772,
                pool_row=43,
                tail=True,
                num_blocks=74157,
                past=2**33,
                context_lens=[1048576, 777],
            ),
        ),
    )

    note = """
    CuTe DSL paged FP8 MQA indexer logits kernel. Covers 32K and 1M decode graphs (with
    padding dummies), mixed steps, DSpark verify steps, and V2 manager paged pools.
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
            q,
            kv_fused,
            weights,
            context_lens,
            block_table,
            schedule_meta,
            max_context_len,
            num_epi_subtiles,
            epi_dtype,
            acc_dtype,
            output_dtype,
        )

    @staticmethod
    def unpack_cache(kv_fused: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(k [num_pages, page, D] uint8 codes, k_scale [num_pages, page] fp32)`.

        A page stores its tokens' data first, then their scales. The codes stay
        uint8 so they can be gathered by page; view them as e4m3 afterwards.
        """
        pages, page = kv_fused.shape[0], kv_fused.shape[1]
        flat = kv_fused.view(torch.uint8).reshape(pages, -1)
        k = flat[:, : page * head_dim].reshape(pages, page, head_dim)
        k_scale = flat[:, page * head_dim : page * (head_dim + 4)].contiguous()
        return k, k_scale.view(torch.float32).reshape(pages, page)

    def reference(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """The causal scores in fp64, rounded once; NaN where unspecified.

        Each sequence's pages are gathered from `kv_fused` first, so a strided
        or offset view and a pool of any size cost only what the sequence
        reads. `schedule_meta` and the three dtype/tiling knobs do not change
        the value and are not read.
        """
        batch, next_n, heads, head_dim = q.shape
        page = kv_fused.shape[1]
        out = torch.full(
            (batch * next_n, max_context_len), float("nan"), dtype=torch.float64, device=q.device
        )
        lens = context_lens.tolist()
        for b in range(batch):
            n = lens[b]
            if n <= 0:
                continue
            pages = block_table[b, : (n + page - 1) // page].long()
            k_pages, s_pages = self.unpack_cache(kv_fused[pages], head_dim)
            codes = k_pages.reshape(-1, head_dim)[:n].contiguous()
            k = codes.view(torch.float8_e4m3fn).float().double()
            s = s_pages.reshape(-1)[:n].double()
            for t in range(next_n):
                row = b * next_n + t
                end = n - next_n + t + 1
                if end <= 0:
                    continue
                qd = q[b, t].float().double()
                dots = (qd @ k[:end].t()).relu()
                w = weights[row].double().unsqueeze(1)
                out[row, :end] = (dots * w).sum(0) * s[:end]
        return out.float()

    def is_valid(
        self,
        q: torch.Tensor,
        kv_fused: torch.Tensor,
        weights: torch.Tensor,
        context_lens: torch.Tensor,
        block_table: torch.Tensor,
        schedule_meta: torch.Tensor,
        max_context_len: int,
        num_epi_subtiles: int = 1,
        epi_dtype: torch.dtype = torch.float32,
        acc_dtype: torch.dtype = torch.float32,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        # One schedule row per CTA plus one, and the op launches one CTA per
        # SM. The schedule's producer takes its own SM count, so the two can
        # disagree. Measured on 16 sequences of 9216 keys: a schedule built for
        # twice the SMs leaves 69632 of the 147456 causal logits unwritten
        # without raising (a light batch can fit the first SMs' partitions and
        # come out right); one built for 8 fewer SMs faults the context with an
        # illegal memory access.
        expected = torch.cuda.get_device_properties(q.device).multi_processor_count + 1
        assert tuple(schedule_meta.shape) == (expected, 2), (
            f"schedule_meta must be [{expected}, 2] (SM count + 1 rows); got "
            f"{tuple(schedule_meta.shape)}"
        )
        # The output has `max_context_len` columns; the scores of a longer
        # context's last keys land in the row's padding and are lost without
        # an error. Reading the lengths is a host sync, which a CUDA-graph
        # capture cannot take: under capture only the schedule is checked.
        if context_lens.numel() and not torch.cuda.is_current_stream_capturing():
            assert int(context_lens.max()) <= max_context_len, (
                f"a context of {int(context_lens.max())} keys exceeds max_context_len "
                f"{max_context_len}"
            )


cute_dsl_fp8_paged_mqa_logits = CuteDslFp8PagedMqaLogits()
