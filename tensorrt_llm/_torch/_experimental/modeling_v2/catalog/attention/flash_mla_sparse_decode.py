# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MQA of one query row per sequence over an fp8_ds_mla pool: FlashMLA's sm100f
sparse fp8 decode kernel."""

from __future__ import annotations

import torch

from tensorrt_llm.flash_mla import flash_mla_with_kvcache as _flash_mla_with_kvcache

from .._op import Arch, Cell, OpWrapper, true_fp32_matmul

ROW_BYTES = 656
D_NOPE = 512
D_ROPE = 64
TILE = 128
#: The query head counts FlashMLA's sm100 sparse decode runs (64, or 64 twice).
KERNEL_HEADS = (64, 128)
#: The most rows (one-row sequences) one call takes: the largest the cells drive. FlashMLA's
#: scheduler exits the process past 11622 on sm_103 (the note's Hard limits).
MAX_ROWS = 8192

#: The tp4 target at the 1M envelope: 16 live heads of the 64-head scratch, sequences to
#: 1048576 keys, the fp8_ds_mla pool's 32-token pages of 63 KV-owner rows per block.
_ML4_1M = dict(h_q=64, h_out=16, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: The tep4 target at the 1M envelope: 16 live heads of its 64-head scratch, top-2048
#: selections, sequences to 1048576 keys, 32-token pages of 63 KV-owner rows per block.
_TEP4_1M = dict(h_q=64, h_out=16, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: The dep4 target at the 1M envelope: all 64 heads of a rank (its scratch is the kernel's
#: 64 heads, no padding), top-2048 selections, sequences to 1048576 keys, 32-token pages of
#: 63 KV-owner rows per block.
_DEP4_1M = dict(h_q=64, h_out=64, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: `pages` (blocks x 63) of a 74183-block pool (the delivered 1M tp4 boot's, above the
#: 72757 blocks a 1M tep4 rank's boot measured and the 64292 the 1M dep4 boot measured per
#: rank at fraction 0.9, on fp8_ds_mla), of the 107917-block far end (the far cells of the
#: 1M attention entries), of the 1100-block pool the dep4 pre-merge eager and context cells
#: spread 1M sequences over, and of the 2048-block pool the tep4, dep4 and tep8 pre-merge
#: cells read past 2**31 bytes.
_BOOT_PAGES = 74183 * 63
_FAR_PAGES = 107917 * 63
_SMALL_PAGES = 1100 * 63
_PRE_PAGES = 2048 * 63
#: The batches the engine captures (padded): an attention-DP rank replays the largest rank's.
_CAPTURED = (1, 2, 4, 8, 16)
#: The pre-merge share of the ragged-tail sweep's row counts: 1..65, either side of every
#: power of two up to 4096, and the top 33.
_TAIL_SHARE = (
    tuple(range(1, 66))
    + tuple(r for k in range(7, 13) for r in (2**k - 1, 2**k, 2**k + 1))
    + tuple(range(8160, 8193))
)
#: The ragged-tail sweep's pre-merge share of the last sub-batch's row counts r: 1..65, 96,
#: 2**k - 1 / 2**k / 2**k + 1 up to 4097, 8160 and 8175..8192.
_RAGGED_SHARE = (
    tuple(range(1, 66))
    + (96,)
    + tuple(r for k in range(7, 14) for r in (2**k - 1, 2**k, 2**k + 1) if r < 8175)
    + (8160,)
    + tuple(range(8175, 8193))
)
#: The dep4 ragged-tail sweep's pre-merge share of the last sub-batch's row counts r: 1..65,
#: 2**k - 1 / 2**k / 2**k + 1 to 4097, r017-land2's 6445, the witness's 8096 and 8175..8192.
_DEP4_TAIL_SHARE = (
    tuple(range(1, 66))
    + tuple(r for k in range(7, 13) for r in (2**k - 1, 2**k, 2**k + 1))
    + (6445, 8096)
    + tuple(range(8175, 8193))
)
#: The tep8 target at the 1M envelope: 8 live heads of its 64-head scratch, top-2048
#: selections, sequences to 1048576 keys, 32-token pages of 63 KV-owner rows per block.
_TEP8_1M = dict(h_q=64, h_out=8, topk=2048, kv_len=1048576, page_size=32, owners=63)
#: `pages` (blocks x 63) of the 147456-block pool bound, the "1M tep8" cells' far pool (a 1M
#: tep8 rank holds 114326 blocks on fp8_ds_mla at fraction 0.9); their pre-merge cells read
#: the 2048-block `_PRE_PAGES` past 2**31 bytes and run the `_RAGGED_SHARE` row counts.
_BOUND_PAGES = 147456 * 63


class FlashMlaSparseDecode(OpWrapper):
    """For every query row r and head h < `out.shape[1]`::

        K[j]   = [e4m3(row_j[:512]) * tile_scale_j (per 128) | bf16(row_j[528:656])]
        p      = softmax_j(softmax_scale * q[r, h] . K[j]),  j in indices[r] (-1 skipped)
        out[r, h] = sum_j p[j] * K[j][:512]

    `kv_cache`: uint8 `[pages, page_size, 1, 656]`, a page's rows contiguous; a slot is
    `page * page_size + offset`. `indices`: int32 `[rows, topk]` slots, -1 padded.
    `q`: bf16 `[rows, h_q, 576]` with h_q 64 or 128 -- the kernel's head counts; a rank
    with fewer heads passes them first and zeros after (the caller's scratch), and
    names how many it keeps through `out`. `sched_meta`: a FlashMLA scheduler object
    (`tensorrt_llm.flash_mla.get_mla_metadata()[0]`), filled on its first call and
    shared by every call of one step with the same rows, h_q and topk. Writes `out`
    (bf16 `[rows, h_out, 512]`, any strides) and returns it; with `out` None, returns the
    kernel's own `[rows, h_q, 512]` output (a view, no copy), whose first heads a caller
    reads in place. At most `MAX_ROWS` rows.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `rows` sequences of one query row, `h_q` kernel heads of which `h_out` are
    # real, `topk` slots over `kv_len`-token sequences in a pool of `pages` pages of
    # `page_size` x `owners` rows; `layers` calls share one sched meta; `captured`
    # replays from a CUDA graph; `holes` scatters -1 through short rows; `view` passes
    # `out` None and gates the returned kernel output.
    # The "1M tp4" cells: `kv_len` is the longest sequence, the others spread below it;
    # `graph_batches` the captured batches (each replay on new slots bitwise equal to an
    # eager call); `far` builds the pool at `pages` / `owners` blocks exactly with the
    # requests' pages its top ones and the first request's last page its top block;
    # `q_offset` makes q and out tail views from that row of a 32768-row step scratch;
    # `context` makes the rows a step's context rows (`chunk_rows` behind `cached`, or the
    # packed `contexts` of (cached, new)), each a one-row sequence, driven in sub-batches of
    # `sub_batch` rows with one sched meta per sub-batch size (`rows`: the largest).
    # `ragged_tail` drives a chunk's last sub-batch at every row count r in 1..`sub_batch`
    # (`tail_rows`: those r only): rows `lead_rows` .. `lead_rows` + r - 1 of the chunk behind
    # `cached`, q and out views from that row of the step's scratch, a sched meta per r.
    # The "1M tep4", "1M dep4" and "1M tep8" cells build the pool on the device, every byte
    # 0xFF (e4m3, fp32 and bf16 NaN) but one owner row of the pages the requests hold.
    # `kv_len` is the longest decode sequence, the others spread below it; `graph_batches`
    # the batches captured (each replay on new queries and new slots in the same pages,
    # bitwise an eager call); `live: 1` with `boot_capture`: each batch captured on the boot
    # warm-up's rows (its longest dummy at kv_len - 1 keys beside one-key fillers), then
    # replayed as attention DP replays it, one live row at kv_len keys behind one-key padding
    # dummies that all read one slot, and an idle rank's one-key dummy behind the same
    # padding;
    # `row_counts`: eager decode steps at each count, a fresh sched meta per step; `far`: the
    # pool is `pages` / `owners` blocks, the requests' pages distinct and its top ones, the
    # first request's last page on the top block; `top_blocks`: the requests' pages drawn
    # from the pool's top that many blocks; `q_offset`: q and out tail views from that row
    # of a 32768-row step scratch; `context`: the rows are a step's context rows
    # (`chunk_rows` behind `cached`, or the packed `contexts` of (cached, new)), each a
    # one-row sequence, in sub-batches of `sub_batch` rows with one sched meta per sub-batch
    # size shared by every layer (`rows`: the largest sub-batch); `ragged_tail`: only a
    # chunk's last sub-batch, each r one call on a sched meta of its own -- a "1M dep4"
    # cell's rows `lead_rows` + [0, r) of the chunk behind `cached`, at every r in
    # 1..`sub_batch` (`tail_rows`: at those r only), a "1M tep4" or "1M tep8" cell's the
    # last sub-batch of a 32768-row chunk behind `cached`, its first r rows at each r in
    # `tails` (every r in 1..`sub_batch` without it; a "1M tep8" cell's the largest first).
    # The test drives the "1M tp4", the "1M tep4", the "1M dep4" and the "1M tep8" cells each
    # with the harness that certified them.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"tep4/tp4 decode batch {b}: 16 real heads padded to 64, graph replay, "
                "3 layers share a sched meta",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * b // 16 + 64,
                    layers=3,
                    captured=True,
                ),
            )
            for b in (1, 2, 4, 8, 16)
        )
        + tuple(
            Cell(
                why=f"dep4 decode batch {b}: 64 heads, graph replay",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=64,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * max(1, b // 16) + 64,
                    layers=2,
                    captured=True,
                ),
            )
            for b in (1, 16, 64)
        )
        + (
            Cell(
                why="short sequences: -1 tails and scattered holes",
                spec=dict(
                    rows=16,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=700,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                    holes=True,
                ),
            ),
            Cell(
                why="NoPE layer: zero query and key rope halves",
                spec=dict(
                    rows=16,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=9000,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                    nope=True,
                ),
            ),
            Cell(
                why="tp8-style 8 real heads padded to 64",
                spec=dict(
                    rows=8,
                    h_q=64,
                    h_out=8,
                    topk=2048,
                    kv_len=4096,
                    page_size=32,
                    owners=63,
                    pages=512,
                    layers=1,
                ),
            ),
        )
        # The cells above at 64-token pages: the same rows, heads, top-k, lengths and graph
        # replays, half the pages for the same token capacity.
        + tuple(
            Cell(
                why=f"64-token pages: tep4/tp4 decode batch {b}: 16 real heads padded to 64, "
                "graph replay, 3 layers share a sched meta",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=32768,
                    page_size=64,
                    owners=63,
                    pages=550 * b // 16 + 32,
                    layers=3,
                    captured=True,
                ),
            )
            for b in (1, 2, 4, 8, 16)
        )
        + tuple(
            Cell(
                why=f"64-token pages: dep4 decode batch {b}: 64 heads, graph replay",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=64,
                    topk=2048,
                    kv_len=32768,
                    page_size=64,
                    owners=63,
                    pages=550 * max(1, b // 16) + 32,
                    layers=2,
                    captured=True,
                ),
            )
            for b in (1, 16, 64)
        )
        + (
            Cell(
                why="64-token pages: short sequences: -1 tails and scattered holes",
                spec=dict(
                    rows=16,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=700,
                    page_size=64,
                    owners=63,
                    pages=256,
                    layers=1,
                    holes=True,
                ),
            ),
            Cell(
                why="64-token pages: NoPE layer: zero query and key rope halves",
                spec=dict(
                    rows=16,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=9000,
                    page_size=64,
                    owners=63,
                    pages=256,
                    layers=1,
                    nope=True,
                ),
            ),
            Cell(
                why="64-token pages: tp8-style 8 real heads padded to 64",
                spec=dict(
                    rows=8,
                    h_q=64,
                    h_out=8,
                    topk=2048,
                    kv_len=4096,
                    page_size=64,
                    owners=63,
                    pages=256,
                    layers=1,
                ),
            ),
        )
        # `out` None, the NVFP4 decode targets' call: the V up-projection reads the live heads
        # of the kernel's 64-head output in place. The decode batches the targets capture,
        # up to a max_batch_size-256 engine's.
        + tuple(
            Cell(
                why=f"out None, tp4/tep4 decode batch {b}: the kernel's 64-head output as "
                "returned, graph replay, 3 layers share a sched meta",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * min(4, max(1, b // 16)) + 64,
                    layers=3,
                    captured=True,
                    view=True,
                ),
            )
            for b in (1, 16, 64, 256)
        )
        + (
            Cell(
                why="out None, dep4 decode batch 64: all 64 heads as returned, graph replay",
                spec=dict(
                    rows=64,
                    h_q=64,
                    h_out=64,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * 4 + 64,
                    layers=2,
                    captured=True,
                    view=True,
                ),
            ),
            Cell(
                why="out None, NoPE layer at 64-token pages: zero query and key rope halves",
                spec=dict(
                    rows=16,
                    h_q=64,
                    h_out=16,
                    topk=2048,
                    kv_len=9000,
                    page_size=64,
                    owners=63,
                    pages=256,
                    layers=1,
                    nope=True,
                    view=True,
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"tep8 decode batch {b}: 8 real heads padded to 64, graph replay, "
                "3 layers share a sched meta",
                spec=dict(
                    rows=b,
                    h_q=64,
                    h_out=8,
                    topk=2048,
                    kv_len=32768,
                    page_size=32,
                    owners=63,
                    pages=1100 * b // 16 + 64,
                    layers=3,
                    captured=True,
                ),
            )
            for b in (1, 2, 4, 8, 16)
        )
        # The tp4 target at the 1M envelope. Decode rows (DecodeTarget._attention, captured at
        # 1/2/4/8/16 padded; the LLM API's default list reaches every batch 1..16) and a mixed
        # step's generation rows (PrefillTarget._attn_gen) are one query each at kv_len - 1;
        # context rows on the decode path (PrefillTarget._attn_ctx on a step whose context KV
        # exceeds 2**19 rows) are a chunk's rows, each one sequence of one row.
        + (
            Cell(
                why="1M tp4 decode graphs at every batch 1..16, boot pool (large): 72 layers share "
                "one sched meta, rows of 1048576 keys down to 1 on the pool's top 73398 blocks "
                "(98.1 GB), the pool's last slot read",
                spec=dict(
                    _ML4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 decode graph 16, far pool (large): 107917 blocks (142.7 GB), the top "
                "slot 217560671 read at byte 142.7e9, 72 layers",
                spec=dict(
                    _ML4_1M,
                    rows=16,
                    pages=_FAR_PAGES,
                    layers=72,
                    captured=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 mixed step's 13 generation rows (large): q and out tail views from row "
                "32755 of the step's 32768-row scratch, eager, a fresh sched meta, boot pool",
                spec=dict(
                    _ML4_1M,
                    rows=13,
                    pages=_BOOT_PAGES,
                    layers=3,
                    q_offset=32755,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 NoPE decode graph 16 (large): layer 35's row 35 as its nine sharers "
                "read it, zero rope halves, boot pool",
                spec=dict(
                    _ML4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=9,
                    captured=True,
                    nope=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 graphs at every batch 1..16 to 2049 keys (2048 of 2049 valid), 72 "
                "layers on one sched meta: the boot-pool cell's captures in pre-merge",
                spec=dict(
                    _ML4_1M,
                    rows=16,
                    kv_len=2049,
                    pages=512,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                ),
            ),
            Cell(
                why="1M tp4 context rows on the decode path (large): the last 32768-row chunk "
                "behind 1015808 cached keys, 4 sub-batches of 8192 one-row sequences, 72 layers",
                spec=dict(
                    _ML4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=32768,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 context rows, ragged chunk (large): 32751 rows behind 1015808, "
                "sub-batches 8192 x 3 and 8175 on two sched metas",
                spec=dict(
                    _ML4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=3,
                    context=True,
                    chunk_rows=32751,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 context rows, packed step (large): chunks of 16384/8192/4096/4096 "
                "behind 1032192/524288/0/2048 on the pool's top 49728 blocks, sub-batches of 8192",
                spec=dict(
                    _ML4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=3,
                    context=True,
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 context rows, 16-row chunk tail (large): positions 1048560..1048575 "
                "in one sub-batch, 72 layers",
                spec=dict(
                    _ML4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=16,
                    cached=1048560,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 context sub-batch at the 8192-row limit: a chunk's last 8192 rows "
                "behind 1040384 cached keys over a 1100-block pool, 2 layers",
                spec=dict(
                    _ML4_1M,
                    rows=8192,
                    pages=1100 * 63,
                    layers=2,
                    context=True,
                    chunk_rows=8192,
                    cached=1040384,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M tp4 mixed step's tail view: 13 generation rows from row 32755 of a "
                "32768-row scratch, keys to 1048576 over a 1100-block pool",
                spec=dict(_ML4_1M, rows=13, pages=1100 * 63, layers=3, q_offset=32755),
            ),
        )
        # The decode-path steps of the envelope prompt's landing (r019-land2: a 1040000-token
        # prompt in 32768-row chunks, the 7-token request packed into its last step) as the
        # target's switch ran them, and the last sub-batch of a chunk at every row count it
        # can take. The kernel reads keys by slot alone, so these run over a 1100-block pool
        # whose pages repeat; the cells above take the slots to the pool's far end.
        + (
            Cell(
                why="1M tp4 r019-land2 last step: 24192 rows behind 1015808 packed with a 7-row "
                "fresh request, sub-batches 8192, 8192, 7815 on two sched metas, 72 layers",
                spec=dict(
                    _ML4_1M,
                    rows=8192,
                    pages=1100 * 63,
                    layers=72,
                    context=True,
                    contexts=((1015808, 24192), (0, 7)),
                    sub_batch=8192,
                ),
            ),
            *(
                Cell(
                    why=f"1M tp4 r019-land2 chunk behind {cached}: 32768 rows in 4 sub-batches "
                    "of 8192 on one sched meta, 2 layers",
                    spec=dict(
                        _ML4_1M,
                        rows=8192,
                        pages=1100 * 63,
                        layers=2,
                        context=True,
                        chunk_rows=32768,
                        cached=cached,
                        sub_batch=8192,
                    ),
                )
                for cached in range(524288, 1015808, 32768)
            ),
            Cell(
                why="1M tp4 r019-land2 ragged tail (large): a chunk's last sub-batch at every r "
                "in 1..8192, rows 24576..24576 + r - 1 behind 1015808, a sched meta each (22 s)",
                spec=dict(
                    _ML4_1M,
                    rows=8192,
                    pages=1100 * 63,
                    layers=1,
                    context=True,
                    cached=1015808,
                    sub_batch=8192,
                    lead_rows=24576,
                    ragged_tail=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tp4 r019-land2 ragged tail, pre-merge share: r in 1..65, 2**k - 1, 2**k, "
                "2**k + 1 for k 7..12, and 8160..8192",
                spec=dict(
                    _ML4_1M,
                    rows=8192,
                    pages=1100 * 63,
                    layers=1,
                    context=True,
                    cached=1015808,
                    sub_batch=8192,
                    lead_rows=24576,
                    tail_rows=_TAIL_SHARE,
                ),
            ),
        )
        # The tep4 target at the 1M envelope, its three call sites: DecodeTarget._attention
        # (captured at 1/2/4/8/16 padded, every batch 1..16 under the LLM API's default list,
        # 72 calls on one sched meta made per step), PrefillTarget._attn_gen (a mixed step's
        # 1..16 generation rows, eager, q and out tail views of the step's scratch, a fresh
        # sched meta), and PrefillTarget._attn_ctx on a step whose context KV exceeds 2**19
        # rows (every context row a one-row sequence, sub-batches of at most 8192 rows, the
        # last one tc mod 8192 rows: any r in 1..8192).
        + (
            Cell(
                why="1M tep4 decode graphs, every batch 1..16, boot pool (large): 72 layers on "
                "one sched meta, rows of 1048576 keys down to 1 in the top 73898 of 74183 "
                "blocks (98.1 GB), the pool's last slot read",
                spec=dict(
                    _TEP4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 decode graph 16, far pool (large): 107917 blocks (142.7 GB), slot "
                "217560671 at byte 1.427e11 read, 72 layers",
                spec=dict(
                    _TEP4_1M,
                    rows=16,
                    pages=_FAR_PAGES,
                    layers=72,
                    captured=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 mixed-step generation tail (large): 13 rows, q and out tail views "
                "from row 32755 of the step's 32768-row scratch, eager, a fresh sched meta, "
                "boot pool",
                spec=dict(
                    _TEP4_1M,
                    rows=13,
                    pages=_BOOT_PAGES,
                    layers=3,
                    q_offset=32755,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 NoPE decode graph 16 (large): owner row 35 as layer 35 and its nine "
                "sharers read it, zero rope halves, boot pool, 9 layers",
                spec=dict(
                    _TEP4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=9,
                    captured=True,
                    nope=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 pre-merge decode graphs at every batch 1..16 to 2049 keys (2048 of "
                "2049 valid, -1 tails below 2048), 72 layers on one sched meta",
                spec=dict(
                    _TEP4_1M,
                    rows=16,
                    kv_len=2049,
                    pages=512,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                ),
            ),
            Cell(
                why="1M tep4 context rows, full chunk (large): 32768 rows behind 1015808 cached "
                "keys, 4 sub-batches of 8192 one-row sequences on one sched meta, 72 layers, "
                "boot pool",
                spec=dict(
                    _TEP4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=32768,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 context rows, ragged chunk (large): 32751 rows behind 1015808, "
                "sub-batches 8192 x 3 and 8175 on two sched metas, boot pool",
                spec=dict(
                    _TEP4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=3,
                    context=True,
                    chunk_rows=32751,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 context rows, packed step (large): chunks of 16384/8192/4096/4096 "
                "behind 1032192/524288/0/2048, sub-batches across request edges, boot pool",
                spec=dict(
                    _TEP4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=3,
                    context=True,
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 context rows, 16-row chunk tail (large): positions 1048560..1048575 "
                "in one sub-batch, 72 layers, boot pool",
                spec=dict(
                    _TEP4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=16,
                    cached=1048560,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 pre-merge context sub-batches past 2**31 pool bytes: a chunk's last "
                "12000 rows behind 1036576 cached keys, 8192 + 3808 on two sched metas, every "
                "slot above 3273604, 2 layers",
                spec=dict(
                    _TEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=2,
                    context=True,
                    chunk_rows=12000,
                    cached=1036576,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M tep4 pre-merge generation tail past 2**31 pool bytes: 13 rows of keys to "
                "1048576, q and out tail views from row 32755 of a 32768-row scratch, 3 layers",
                spec=dict(
                    _TEP4_1M,
                    rows=13,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=3,
                    q_offset=32755,
                ),
            ),
            Cell(
                why="1M tep4 pre-merge r019-land1 last step: 24672 context rows behind 1015328 "
                "cached keys, sub-batches 8192 x 3 + 96 on two sched metas, 72 layers, every "
                "slot above 3273604",
                spec=dict(
                    _TEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=72,
                    context=True,
                    chunk_rows=24672,
                    cached=1015328,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M tep4 context rows, ragged last sub-batch at every r in 1..8192 (large): "
                "rows 24576 + [0, r) of a 32768-row chunk behind 1015328 cached keys (positions "
                "to 1048095), each r one call on a sched meta of its own, every slot above "
                "3273604",
                spec=dict(
                    _TEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=1,
                    context=True,
                    cached=1015328,
                    sub_batch=8192,
                    ragged_tail=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep4 pre-merge ragged last sub-batch, the sweep's share: r in 1..65, 96, "
                "2**k - 1 / 2**k / 2**k + 1 to 4097, 8160 and 8175..8192 behind 1015328 cached "
                "keys",
                spec=dict(
                    _TEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=1,
                    context=True,
                    cached=1015328,
                    sub_batch=8192,
                    ragged_tail=True,
                    tails=_RAGGED_SHARE,
                ),
            ),
        )
        # The dep4 target at the 1M envelope, its three call sites: DecodeTarget._attention
        # (captured at 1/2/4/8/16 padded, every batch 1..16 under the LLM API's default list,
        # 72 calls on one sched meta made per step; under attention DP every rank replays the
        # largest rank's batch, one live row behind one-key padding dummies; eager at any
        # count 1..16 beside a prefilling peer), PrefillTarget._attn_gen (a mixed step's 1..15
        # generation rows, eager, q and out tail views of the step's scratch, a fresh sched
        # meta), and PrefillTarget._attn_ctx on a step whose context KV exceeds 2**19 rows
        # (every context row a one-row sequence, sub-batches of at most 8192 rows, the last one
        # tc - 8192 x (ceil(tc / 8192) - 1) rows: any r in 1..8192).
        + (
            Cell(
                why="1M dep4 decode graphs, every batch 1..16, boot pool (large): 72 layers on "
                "one sched meta, rows of 1048576 keys down to 1 in the top 73898 of 74183 "
                "blocks (98.1 GB), the pool's last slot read",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 ADP boot capture at 1/2/4/8/16, boot pool (large): captured on "
                "the warm-up's 1048575-key longest dummy beside one-key fillers, replayed with "
                "one live row of 1048576 keys behind one-key padding dummies on one slot, then "
                "an idle rank's dummy behind them, 72 layers",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=72,
                    captured=True,
                    graph_batches=_CAPTURED,
                    live=1,
                    boot_capture=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 decode graph 16, far pool (large): 107917 blocks (142.7 GB), slot "
                "217560671 at byte 1.427e11 read, 72 layers",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    pages=_FAR_PAGES,
                    layers=72,
                    captured=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 mixed-step generation tail (large): 13 rows, q and out tail views "
                "from row 32755 of the step's 32768-row scratch, eager, a fresh sched meta, "
                "boot pool",
                spec=dict(
                    _DEP4_1M,
                    rows=13,
                    pages=_BOOT_PAGES,
                    layers=3,
                    q_offset=32755,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 NoPE decode graph 16 (large): owner row 35 as layer 35 and its "
                "sharers read it, zero rope halves, boot pool, 9 layers",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=9,
                    captured=True,
                    nope=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 pre-merge decode graphs at every batch 1..16 to 2049 keys (2048 of "
                "2049 valid, -1 tails below 2048), 72 layers on one sched meta",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    kv_len=2049,
                    pages=512,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                ),
            ),
            Cell(
                why="1M dep4 pre-merge eager decode steps at every row count 1..16 beside a "
                "prefilling peer: a fresh sched meta each, 72 layers, keys to 1048576 over an "
                "1100-block pool",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    pages=_SMALL_PAGES,
                    layers=72,
                    row_counts=tuple(range(1, 17)),
                ),
            ),
            Cell(
                why="1M dep4 context rows, full chunk (large): 32768 rows behind 1015808 cached "
                "keys, 4 sub-batches of 8192 one-row sequences on one sched meta, 72 layers, "
                "boot pool",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=32768,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 context rows, ragged chunk (large): 32751 rows behind 1015808, "
                "sub-batches 8192 x 3 and 8175 on two sched metas, boot pool",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=3,
                    context=True,
                    chunk_rows=32751,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 context rows, packed step (large): chunks of 16384/8192/4096/4096 "
                "behind 1032192/524288/0/2048, sub-batches across request edges, boot pool",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_BOOT_PAGES,
                    layers=3,
                    context=True,
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 context rows, 16-row chunk tail (large): positions 1048560..1048575 "
                "in one sub-batch, 72 layers, boot pool",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    pages=_BOOT_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=16,
                    cached=1048560,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 pre-merge context rows, one 8192-row sub-batch: the last chunk of a "
                "1M prompt behind 1040384 cached keys over an 1100-block pool, 2 layers",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_SMALL_PAGES,
                    layers=2,
                    context=True,
                    chunk_rows=8192,
                    cached=1040384,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M dep4 pre-merge context sub-batches past 2**31 pool bytes: a chunk's last "
                "12000 rows behind 1036576 cached keys, 8192 + 3808 on two sched metas, every "
                "slot above 3322368, 2 layers",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=2,
                    context=True,
                    chunk_rows=12000,
                    cached=1036576,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M dep4 pre-merge ADP boot capture past 2**31 pool bytes: graphs 1/2/4/8/16 "
                "captured on the 1048575-key dummy and one-key fillers, replayed with one live "
                "row behind padding on one slot and with an idle rank's dummy, 72 layers",
                spec=dict(
                    _DEP4_1M,
                    rows=16,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=72,
                    captured=True,
                    graph_batches=_CAPTURED,
                    live=1,
                    boot_capture=True,
                ),
            ),
            Cell(
                why="1M dep4 pre-merge generation tail past 2**31 pool bytes: 13 rows of keys to "
                "1048576, q and out tail views from row 32755 of a 32768-row scratch, 3 layers",
                spec=dict(
                    _DEP4_1M,
                    rows=13,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=3,
                    q_offset=32755,
                ),
            ),
            Cell(
                why="1M dep4 pre-merge r017-land2 last step: 22829 context rows behind 1015808 "
                "cached keys (positions to 1038636), sub-batches 8192 x 2 + 6445 on two sched "
                "metas, 72 layers, every slot above 3322368",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=72,
                    context=True,
                    chunk_rows=22829,
                    cached=1015808,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M dep4 context rows, ragged last sub-batch at every r in 1..8192 (large): "
                "rows 16384 + [0, r) of a chunk behind 1015808 cached keys (positions to "
                "1040383), each r one call on a sched meta of its own, every slot above 3322368",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=1,
                    context=True,
                    cached=1015808,
                    lead_rows=16384,
                    sub_batch=8192,
                    ragged_tail=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M dep4 pre-merge ragged last sub-batch, the sweep's share: r in 1..65, "
                "2**k - 1 / 2**k / 2**k + 1 to 4097, 6445, 8096 and 8175..8192, rows 16384 + "
                "[0, r) behind 1015808 cached keys",
                spec=dict(
                    _DEP4_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=1,
                    context=True,
                    cached=1015808,
                    lead_rows=16384,
                    sub_batch=8192,
                    tail_rows=_DEP4_TAIL_SHARE,
                ),
            ),
        )
        # The tep8 target at the 1M envelope, its three call sites: DecodeTarget._attention
        # (captured at 1/2/4/8/16 padded, every batch 1..16 under the LLM API's default list,
        # 72 calls on one sched meta made per step), PrefillTarget._attn_gen (a mixed step's
        # 1..16 generation rows, eager, q and out tail views of the step's scratch, a fresh
        # sched meta), and PrefillTarget._attn_ctx on a step whose context KV exceeds 2**19
        # rows (every context row a one-row sequence, sub-batches of at most 8192 rows, the
        # last one tc - 8192 x (ceil(tc / 8192) - 1) rows: any r in 1..8192).
        + (
            Cell(
                why="1M tep8 decode graphs, every batch 1..16, far pool (large): 72 layers on "
                "one sched meta, rows of 1048576 keys down to 1 in the top 73898 of 147456 "
                "blocks (195.0 GB), slot 297271295 at byte 1.950e11 read",
                spec=dict(
                    _TEP8_1M,
                    rows=16,
                    pages=_BOUND_PAGES,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 mixed-step generation tail (large): 13 rows, q and out tail views "
                "from row 32755 of the step's 32768-row scratch, eager, a fresh sched meta, "
                "far pool",
                spec=dict(
                    _TEP8_1M,
                    rows=13,
                    pages=_BOUND_PAGES,
                    layers=3,
                    q_offset=32755,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 NoPE decode graph 16 (large): owner row 35, which layer 35 and its "
                "nine sharers read, zero rope halves, far pool, 9 layers",
                spec=dict(
                    _TEP8_1M,
                    rows=16,
                    pages=_BOUND_PAGES,
                    layers=9,
                    captured=True,
                    nope=True,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 pre-merge decode graphs at every batch 1..16 to 2049 keys (2048 of "
                "2049 valid, -1 tails below 2048), 72 layers on one sched meta",
                spec=dict(
                    _TEP8_1M,
                    rows=16,
                    kv_len=2049,
                    pages=512,
                    layers=72,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                ),
            ),
            Cell(
                why="1M tep8 context rows, full chunk (large): 32768 rows behind 1015808 cached "
                "keys, 4 sub-batches of 8192 one-row sequences on one sched meta, 72 layers, "
                "far pool",
                spec=dict(
                    _TEP8_1M,
                    rows=8192,
                    pages=_BOUND_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=32768,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 context rows, ragged chunk (large): 32751 rows behind 1015808, "
                "sub-batches 8192 x 3 and 8175 on two sched metas, far pool",
                spec=dict(
                    _TEP8_1M,
                    rows=8192,
                    pages=_BOUND_PAGES,
                    layers=3,
                    context=True,
                    chunk_rows=32751,
                    cached=1015808,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 context rows, packed step (large): chunks of 16384/8192/4096/4096 "
                "behind 1032192/524288/0/2048, sub-batches across request edges, far pool",
                spec=dict(
                    _TEP8_1M,
                    rows=8192,
                    pages=_BOUND_PAGES,
                    layers=3,
                    context=True,
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 context rows, 16-row chunk tail (large): positions 1048560..1048575 "
                "in one sub-batch, 72 layers, far pool",
                spec=dict(
                    _TEP8_1M,
                    rows=16,
                    pages=_BOUND_PAGES,
                    layers=72,
                    context=True,
                    chunk_rows=16,
                    cached=1048560,
                    sub_batch=8192,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 pre-merge context sub-batches past 2**31 pool bytes: a chunk's last "
                "12000 rows behind 1036576 cached keys, 8192 + 3808 on two sched metas, every "
                "slot above 3273604, 2 layers",
                spec=dict(
                    _TEP8_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=2,
                    context=True,
                    chunk_rows=12000,
                    cached=1036576,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M tep8 pre-merge generation tail past 2**31 pool bytes: 13 rows of keys to "
                "1048576, q and out tail views from row 32755 of a 32768-row scratch, 3 layers",
                spec=dict(
                    _TEP8_1M,
                    rows=13,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=3,
                    q_offset=32755,
                ),
            ),
            Cell(
                why="1M tep8 pre-merge r018-land1 last step: 24672 context rows behind 1015328 "
                "cached keys, sub-batches 8192 x 3 + 96 on two sched metas, 72 layers, every "
                "slot above 3273604",
                spec=dict(
                    _TEP8_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=72,
                    context=True,
                    chunk_rows=24672,
                    cached=1015328,
                    sub_batch=8192,
                ),
            ),
            Cell(
                why="1M tep8 context rows, ragged last sub-batch at every r in 1..8192 (large): "
                "rows 24576 + [0, r) of a 32768-row chunk behind 1015328 cached keys (positions "
                "to 1048095), each r one call on a sched meta of its own, every slot above "
                "3273604",
                spec=dict(
                    _TEP8_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=1,
                    context=True,
                    cached=1015328,
                    sub_batch=8192,
                    ragged_tail=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M tep8 pre-merge ragged last sub-batch, the sweep's share: r in 1..65, 96, "
                "2**k - 1 / 2**k / 2**k + 1 to 4097, 8160 and 8175..8192 behind 1015328 cached "
                "keys",
                spec=dict(
                    _TEP8_1M,
                    rows=8192,
                    pages=_PRE_PAGES,
                    top_blocks=400,
                    layers=1,
                    context=True,
                    cached=1015328,
                    sub_batch=8192,
                    ragged_tail=True,
                    tails=_RAGGED_SHARE,
                ),
            ),
        )
    )
    # `out` None at the 1M config, the decode call the targets read in place (dep4's always,
    # tp4 / tep4's under `_GEN_ATTN_FLASH_MLA` with the fused glue): the "1M tep4" / "1M dep4"
    # harness's pools and rows, every layer's call returning the kernel's 64-head output, its
    # live heads bitwise the `out` call's copy on the same inputs and gated to the reference.
    CELLS += (
        Cell(
            why="1M view tp4/tep4 decode graphs at every batch 1..16, boot pool (large): out "
            "None, the 16 live heads of the 64-head output read in place, 3 layers on one "
            "sched meta, rows of 1048576 keys down to 1, the pool's last slot read",
            spec=dict(
                _TEP4_1M,
                rows=16,
                pages=_BOOT_PAGES,
                layers=3,
                captured=True,
                graph_batches=tuple(range(1, 17)),
                far=True,
                view=True,
                large=True,
            ),
        ),
        Cell(
            why="1M view tp4/tep4 decode graph 16, far pool (large): 107917 blocks (142.7 GB), "
            "out None, the top slot read at byte 1.427e11",
            spec=dict(
                _TEP4_1M,
                rows=16,
                pages=_FAR_PAGES,
                layers=3,
                captured=True,
                far=True,
                view=True,
                large=True,
            ),
        ),
        Cell(
            why="1M view dep4 decode graphs at every batch 1..16, boot pool (large): out None, "
            "all 64 heads as returned, 3 layers on one sched meta",
            spec=dict(
                _DEP4_1M,
                rows=16,
                pages=_BOOT_PAGES,
                layers=3,
                captured=True,
                graph_batches=tuple(range(1, 17)),
                far=True,
                view=True,
                large=True,
            ),
        ),
        Cell(
            why="1M view dep4 NoPE decode graph 16 (large): owner row 35, zero rope halves, out "
            "None, boot pool",
            spec=dict(
                _DEP4_1M,
                rows=16,
                pages=_BOOT_PAGES,
                layers=3,
                captured=True,
                nope=True,
                far=True,
                view=True,
                large=True,
            ),
        ),
    )

    note = """
    Call sites: mistral_large4 targets, the generation rows of every layer. Kernels
    (sm_103): smxx get_mla_metadata (first call of a step), sm100 head64
    flash_fwd_splitkv_mla_fp8_sparse (V32 format), the split combine. The kernel holds
    each tile scale as E8M0 rounding toward zero: exact for the power-of-two scales the
    fp8_ds_mla writer stores, off by up to one binade for any other.

    `out` None launches the same kernels on the same arguments and returns their output
    instead of copying its first `h_out` heads (an exact copy): the live heads are bitwise
    the `out` call's (tested), and a V up-projection BMM over the strided view returns the
    bits it returns over the copy (tested). Nothing in it depends on the sequence length or
    the pool: its cells run the 32K decode batches and the 1M config's (`1M view`, keys to
    1048576 on the boot and far pools, 16 and 64 live heads).

    Cost: a call that makes a fresh sched meta pays a fixed cost over one that shares the
    step's sched meta.

    The "1M tp4" cells are the Mistral Large 4 NVFP4 tp4 target at max_seq_len 1048576 with
    chunked prefill (32768-row steps, up to 16 requests), 16 live heads of its 64-head
    scratch, `kv_cache` the whole latent pool viewed [blocks * 63, 32, 1, 656] (FlashMLA
    requires the cache contiguously valid from its first byte to its last on sm100),
    slots (block * 63 + latent_pool_row) * 32 + offset. Three call sites: the decode step
    (captured, 72 calls on one sched meta made per step), a mixed step's generation rows
    (eager, q and out tail views of the step's scratch, a fresh sched meta), and the
    context rows of a step whose context KV exceeds 2**19 rows, each row a one-row
    sequence, in sub-batches of at most 8192 rows with one sched meta per sub-batch size
    shared by the 72 layers. The kernel reads keys by slot alone: a 1M sequence reaches it
    only as slot magnitude, which the cells take to the pool's last slot. Each graph replay
    in the cells matches an eager call bitwise.

    The "r019-land2" cells are the decode-path steps of a 1040000-token prompt landing at
    the envelope, as the target's switch ran them: 15 chunks of 32768 rows behind 524288 ..
    983040 cached keys, 4 sub-batches of 8192 each, and the last step, 24192 rows behind
    1015808 packed with a 7-token request (sub-batches 8192, 8192, 7815). A step's last
    sub-batch is r = (its context rows mod 8192) rows, any r in 1..8192 (generation rows
    behind a chunk, any prompt's last chunk): the ragged-tail cell drives every r on a sched
    meta of its own, and the scheduler took each. The split-KV schedule follows the row
    count, so a ragged sub-batch's rows are not bitwise a full one's (on one draw, 6857 of
    the 8192 r differ from the 8192-row call on up to 67 rows). Every r lands inside the
    band: per-(row, head) rel L2 at most 5.33e-3 and whole call 2.55e-4 on the cell's draw,
    5.51e-3 and 2.44e-4 on another. The sweep's 8192 kernel calls make it large; pre-merge
    drives r in 1..65, either side of each power of two to 4096, and 8160..8192.

    The "1M tep4" cells are the Mistral Large 4 NVFP4 tep4 target at max_seq_len 1048576 with
    chunked prefill (steps of up to 32768 rows and 16 requests), 16 live heads of its
    64-head scratch, `kv_cache` the whole latent pool viewed [blocks * 63, 32, 1, 656]
    (FlashMLA requires the cache contiguously valid from its first byte to its last on
    sm100), slots (block * 63 + latent_pool_row) * 32 + offset. Its decode step, a mixed
    step's generation rows, and the context rows of a step whose context KV exceeds 2**19
    rows, each row then a one-row sequence in sub-batches of at most 8192 rows with one
    sched meta per sub-batch size shared by the 72 layers. The kernel reads keys by slot
    alone, so a 1M sequence reaches it as slot magnitude: the cells read the pool's top
    blocks. Every graph replay matches an eager call bitwise.

    The decode path's last sub-batch is tc mod 8192 rows, any r in 1..8192, and FlashMLA's
    scheduler splits some rows of a call over their keys (`num_splits` > 1, the parts merged
    in fp32 by the combine kernel), which ones depending on r and the device's sm parts
    alone. Swept at every r in 1..8192 on a sched meta each (rows 24576 + [0, r) of a chunk
    behind 1015328 keys, positions to 1048095), one process: every r ran, every row finite
    and in the band (one seed: per (row, head) at most 5.25e-3, whole call 2.40e-4), every row
    the scheduler kept in one part bitwise the 8192-row call's (which splits none). Rows
    split per call on sm_103 (one sm part per SM): every row at r 1..127 (all 96 of r019's
    96-row call), none at 1334 r (128..152, 4096, 8184..8192 among them), 71 to 147 at the
    other 6731 (121 at 8175); 828080 split rows over the sweep, about 40% of them not bitwise
    the 8192-row call's (327794 and 327350 at two seeds). Its 8192 calls took 18.2 s, so the
    sweep is large and a pre-merge share runs the 103 row counts of `_RAGGED_SHARE`.

    The "1M dep4" cells are the Mistral Large 4 NVFP4 dep4 target at max_seq_len 1048576 with
    chunked prefill (steps of up to 32768 rows and 16 requests per rank) under attention DP:
    a rank calls with all 64 heads of its own rows (q its [*, 64, 576] scratch, no padding
    heads; out [*, 64, 512]), `kv_cache` the whole latent pool viewed [blocks * 63, 32, 1,
    656] (FlashMLA requires the cache contiguously valid from its first byte to its last on
    sm100), slots (block * 63 + latent_pool_row) * 32 + offset. Its decode step: graphs
    captured at 1/2/4/8/16 on the boot warm-up's rows (the 1048575-key longest dummy beside
    one-key fillers) and replayed as attention DP pads them, one live row behind one-key
    padding dummies whose rows repeat one slot and one query, or an idle rank's one-key dummy
    behind them; graphs at every batch 1..16; eager steps at every count 1..16 beside a
    prefilling peer, a fresh sched meta each. A mixed step's generation rows, tail views of
    the step's scratch. The context rows of a step whose context KV exceeds 2**19 rows, each
    then a one-row sequence in sub-batches of at most 8192 rows with one sched meta per
    sub-batch size shared by the 72 layers. Every graph replay matches an eager call
    bitwise. The kernel reads keys by slot alone, so a 1M sequence reaches it as slot
    magnitude: the cells read the pool's top blocks.

    The decode path's last sub-batch is tc - 8192 x (ceil(tc / 8192) - 1) rows, any r in
    1..8192 (r017-land2's 1,038,637-token prompt ended on 6445). FlashMLA's scheduler splits
    some rows of a call over their keys (`num_splits` > 1, the parts merged in fp32 by the
    combine kernel), which ones depending on r and the device's sm parts alone. Swept at the
    dep4 call at every r in 1..8192, largest first, one process, a sched meta each (rows
    16384 + [0, r) of a chunk behind 1015808 keys, positions to 1040383, q, slots and out
    views of the step's buffers): every r ran, every row finite and in the band (two seeds:
    per (row, head) at most 5.65e-3 and 6.03e-3, a call's rel L2 at most 1.92e-4 and
    1.93e-4), every row neither that call nor the 8192-row call split bitwise the 8192-row
    call's (which splits none). Rows split per call on sm_103 (one sm part per SM), the
    same at both seeds: every row at r 1..127, none at 1334 r (128..152, 4096, 8184..8192
    among them), 71 to 147 at the other 6731 (139 at 6445, 130 at 8096, 121 at 8175); 828080
    split rows over the sweep, 675860 and 680683 of them not bitwise the 8192-row call's at
    the two seeds. Its 8192 calls took 19.1 s and the cell 25.7 s, so the sweep is large and
    a pre-merge share runs the 103 row counts of `_DEP4_TAIL_SHARE`; largest first, the sweep
    peaked at 9.44 GiB allocated, 11.10 GiB reserved.

    The "1M tep8" cells are the Mistral Large 4 NVFP4 tep8 target at max_seq_len 1048576 with
    chunked prefill (steps of up to 32768 rows and 16 requests), 8 live heads of its
    64-head scratch, `kv_cache` the whole latent pool viewed [blocks * 63, 32, 1, 656]
    (FlashMLA requires the cache contiguously valid from its first byte to its last on
    sm100), slots (block * 63 + latent_pool_row) * 32 + offset. It calls the op for its
    decode step, a mixed step's generation rows, and the context rows of a step whose
    context KV exceeds 2**19 rows, each context row then a one-row sequence in sub-batches
    of at most 8192 rows with one sched meta per sub-batch size shared by the 72 layers.
    The kernel reads keys by slot alone, so a 1M sequence reaches it as slot magnitude: the
    cells read the top blocks of a pool at the 147456-block bound, above the 114326 blocks
    (top slot 230,481,215) a 1M tep8 rank holds. Every graph replay matches an eager call
    bitwise. A 147456-block cell peaks at 182.8-185.9 GiB of device memory (its pool 181.6
    GiB).

    At the tep8 call the decode path's last sub-batch is likewise tc - 8192 x (ceil(tc /
    8192) - 1) rows, any r in 1..8192: get_mla_metadata deals the call's 37 r blocks (per
    row 32 of 64 keys and 5 of overhead) in row order to the device's sm parts, ceil(37 r /
    parts) + 5 each, so which rows split depends on r and the SM count alone. Swept at
    every r in 1..8192 on a sched meta each (rows 24576 + [0, r) of a chunk behind 1015328
    keys, positions to 1048095, heads 8..63 zero), one process: every r ran, every row
    finite and in the band (one draw: per (row, head) at most 5.12e-3, whole call
    2.82e-4), every row neither scheduler split bitwise the 8192-row call's (which splits
    none). Rows split per call on sm_103 (one sm part per SM): every row at r 1..127
    (all 96 of r018-land1's 96-row call), none at 1334 r (128..152, 4096, 8184..8192 among
    them), 71 to 147 at the other 6731 (121 at 8175, 126 at 8176), r for r the same as at
    16 live heads; 828080 split rows over the sweep, about 22% of them not bitwise the
    8192-row call's (185190 and 180132 at two draws). Its 8192 calls took 17.5 s, so the
    sweep is large and a pre-merge share runs the 103 row counts of `_RAGGED_SHARE`; walked
    largest or smallest r first it reserved 7.91 GiB either way.

    Hard limits (sm_103)
      rows          get_decoding_sched_meta takes 4 * (5 * rows + 1) bytes of dynamic
                    shared memory, against the device's opt-in 232448, which caps one
                    call at 11622 rows; past it the scheduler prints "CUDA error
                    (.../flashmla-src/csrc/smxx/decode/get_decoding_sched_meta/
                    get_decoding_sched_meta.cu:111): invalid argument" and exits the
                    process (rc 1), no exception. Probed one process each, at the tp4
                    and tep4 calls (16 heads out), at the dep4 call (64 heads in and
                    out, top-2048 behind a 1M prefix) and at the tep8 call (8 heads
                    out): 8192 and 11622 rows run (finite, rel L2 2.5e-4 and 2.4e-4 on
                    64 rows at 16 heads, 2.1e-4 and 1.9e-4 at 64, 2.9e-4 and 2.5e-4 at
                    8), 11623 and 16384 exit; every r in 1..8192 ran, one process, a
                    fresh sched meta each. is_valid refuses rows > 8192, the most the
                    cells drive
      indices       int32 slots; the kernel's TMA row coordinate is block * 32 + offset
                    = the slot, in int32 (656-byte rows of a [pages * 32, 656] tensor
                    map): 2**31 slots = 67,108,864 pages = 1,065,220 blocks of 63, not
                    reached (107917 blocks: top slot 217,560,671; 147456 blocks: top
                    slot 297,271,295)
      slot bytes    slot * 656 is the TMA unit's address (64-bit) and the tile scales'
                    load offset is int64; 2**31 bytes at slot 3,273,604, 2**37 at slot
                    209,510,600: probed at the pools' last slots, bytes 98,106,720,112
                    (74183 blocks), 142,719,800,176 (107917 blocks) and
                    195,009,969,520 (147456 blocks), large; in pre-merge every key past
                    2**31 bytes (slots from 3,322,368)                    passed
      indices row   row * topk in int32 (the kernel's indices offset): 2**31 at
                    1,048,576 rows of 2048, past the row cap              not reached
      q / o_accum   rows * 64 * 576 = rows * 36864 q elements pass 2**31 at 58,255
                    rows; the split accumulator (rows + sm parts) * 64 * 512 fp32, its
                    int32 offset n_split * 32768 at rows + sm parts = 65536: both past
                    the row cap                                           not reached
      fp32          no index passes through fp32
      Not certified: more than 8192 rows per call (11622 ran once, in the probe), pools
      past 147456 blocks (past 107917 at 16 and 64 heads out), h_q 128, kv_len past
      1048576, page sizes other than 32 at 1M; which rows a call splits on a device with
      another SM count (the band holds either way).
    """

    def __call__(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        sched_meta,
        softmax_scale: float,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        rows, h_q = q.shape[0], q.shape[1]
        o, _ = _flash_mla_with_kvcache(
            q.view(rows, 1, h_q, q.shape[-1]),
            kv_cache,
            None,
            None,
            D_NOPE,
            sched_meta,
            softmax_scale=softmax_scale,
            is_fp8_kvcache=True,
            indices=indices.view(rows, 1, indices.shape[-1]),
        )
        if out is None:
            return o[:, 0]
        out.copy_(o[:, 0, : out.shape[1]])
        return out

    def reference(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        sched_meta,
        softmax_scale: float,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """fp32, exact dequantization, softmax over the valid slots; a new tensor shaped
        like `out` (`[rows, h_q, 512]` when `out` is None), rounded to bf16 once. Ignores
        `sched_meta`."""
        shape = (q.shape[0], q.shape[1], D_NOPE) if out is None else tuple(out.shape)
        h_out = shape[1]
        flat = kv_cache.view(torch.uint8).reshape(-1, ROW_BYTES)
        res = torch.zeros(shape, dtype=torch.float32, device=q.device)
        with true_fp32_matmul():
            for r in range(q.shape[0]):
                idx = indices[r]
                idx = idx[idx >= 0].long()
                if idx.numel() == 0:
                    continue
                rows = flat[idx]
                codes = rows[:, :D_NOPE].contiguous().view(torch.float8_e4m3fn).float()
                scale = rows[:, D_NOPE : D_NOPE + 16].contiguous().view(torch.float32)
                nope = (codes.view(-1, D_NOPE // TILE, TILE) * scale.unsqueeze(-1)).view(-1, D_NOPE)
                rope = rows[:, D_NOPE + 16 :].contiguous().view(torch.bfloat16).float()
                k = torch.cat([nope, rope], 1)
                s = (q[r, :h_out].float() @ k.T) * softmax_scale
                res[r] = torch.softmax(s, -1) @ nope
        return res.to(torch.bfloat16 if out is None else out.dtype)

    def is_valid(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        sched_meta,
        softmax_scale: float,
        out: torch.Tensor | None,
    ) -> None:
        if q.shape[1] not in KERNEL_HEADS:
            raise ValueError(f"h_q {q.shape[1]} is not one of the kernel's {KERNEL_HEADS}")
        if out is not None and out.shape[1] > q.shape[1]:
            raise ValueError("out names more heads than q carries")
        if indices.dtype != torch.int32:
            raise ValueError("indices must be int32 slots")
        if kv_cache.shape[-1] != ROW_BYTES or kv_cache.dim() != 4 or kv_cache.shape[2] != 1:
            raise ValueError("kv_cache must be [pages, page_size, 1, 656]")
        if q.shape[0] > MAX_ROWS:
            raise ValueError(
                f"{q.shape[0]} rows in one call, more than {MAX_ROWS}: FlashMLA's decode "
                "scheduler exits the process past its shared-memory row limit (11622 rows on "
                "sm_103); split the rows into sub-batches"
            )


flash_mla_sparse_decode = FlashMlaSparseDecode()
