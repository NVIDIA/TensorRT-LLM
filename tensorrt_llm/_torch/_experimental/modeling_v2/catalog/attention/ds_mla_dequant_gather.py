# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather context requests' fp8_ds_mla latent rows out of the paged pool, dequantized to bf16."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper

DIM_NOPE = 512
DIM_ROPE = 64
TILE = 128
ROW_BYTES = 656


@triton.jit
def _gather_kernel(
    pool_ptr,  # uint8 [blocks, owners, tpb, 656]
    owners,
    row,
    kv_req_ptr,  # int32 [N]
    kv_pos_ptr,  # int32 [N]
    block_table_ptr,
    bt_stride,
    out_ptr,  # bf16 [N, 576] (row stride out_stride)
    out_stride,
    TPB: tl.constexpr,
):
    w = tl.program_id(0)
    part = tl.program_id(1)
    req = tl.load(kv_req_ptr + w).to(tl.int64)
    pos = tl.load(kv_pos_ptr + w).to(tl.int64)
    block = tl.load(block_table_ptr + req * bt_stride + pos // TPB).to(tl.int64)
    src = pool_ptr + ((block * owners + row) * TPB + pos % TPB) * 656
    dst = out_ptr + w.to(tl.int64) * out_stride
    if part < 4:
        cols = part * 128 + tl.arange(0, 128)
        codes = tl.load(src + cols).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale = tl.load((src + 512 + part * 4).to(tl.pointer_type(tl.float32)))
        tl.store(dst + cols, (codes * scale).to(tl.bfloat16))
    else:
        r = tl.arange(0, 64)
        rope = tl.load((src + 528).to(tl.pointer_type(tl.bfloat16)) + r)
        tl.store(dst + 512 + r, rope)


class DsMlaDequantGather(OpWrapper):
    """`out[w] = dequant(pool[block_table[kv_req[w], kv_pos[w] // tpb], row, kv_pos[w] % tpb])`::

        out[w, :512]   = e4m3(code) * tile_scale    (exact in bf16: 3 mantissa bits times 2^k)
        out[w, 512:576] = the stored bf16 rope

    The bf16 workspace a dense-key sparse prefill kernel reads: the context requests'
    whole KV (cached prefix and this step's tokens), each request's run contiguous.
    `pool`: uint8 `[blocks, owners, tokens_per_block, 656]`; `kv_req` / `kv_pos`:
    int32 `[N]`; `out`: bf16 `[N, 576]` (row stride free). Returns `out`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: context requests with `cached` + `new` tokens each, a pool of `blocks` x
    # `owners` pages of `tokens_per_block`, owner `row`. The "1M tp4", "1M tep4", "1M dep4"
    # and "1M tep8" cells add `out_rows` (`out` leads a buffer of that many rows, the rest
    # left alone), the block table's `table_rows` (one per request when absent) x
    # `table_width`, `nope` (zero rope: the NoPE source's rows), `far` (the requests' pages
    # are the pool's top ones) and `prefixes` (every N below the cell's is driven too,
    # each bitwise the leading rows of the full call).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one 32768-token request (the unchunked envelope)",
            spec=dict(cached=[0], new=[32768], blocks=1100, owners=63, tokens_per_block=32, row=31),
        ),
        Cell(
            why="three requests, cached prefixes mid-page, the last owner row",
            spec=dict(
                cached=[0, 1029, 5185],
                new=[300, 2047, 1],
                blocks=600,
                owners=63,
                tokens_per_block=32,
                row=62,
            ),
        ),
        Cell(
            why="sixteen 2048-token requests (a full prefill batch)",
            spec=dict(
                cached=[0] * 16, new=[2048] * 16, blocks=1100, owners=63, tokens_per_block=32, row=0
            ),
        ),
        Cell(
            why="64-token pages",
            spec=dict(cached=[37], new=[1000], blocks=64, owners=63, tokens_per_block=64, row=2),
        ),
        # The 32-token cells above at 64-token pages: the same requests and owner rows, half
        # the blocks for the same token capacity.
        Cell(
            why="64-token pages: one 32768-token request (the unchunked envelope)",
            spec=dict(cached=[0], new=[32768], blocks=550, owners=63, tokens_per_block=64, row=31),
        ),
        Cell(
            why="64-token pages: three requests, cached prefixes mid-page, the last owner row",
            spec=dict(
                cached=[0, 1029, 5185],
                new=[300, 2047, 1],
                blocks=300,
                owners=63,
                tokens_per_block=64,
                row=62,
            ),
        ),
        Cell(
            why="64-token pages: sixteen 2048-token requests (a full prefill batch)",
            spec=dict(
                cached=[0] * 16, new=[2048] * 16, blocks=550, owners=63, tokens_per_block=64, row=0
            ),
        ),
        # The tp4 target at the 1M envelope, workspace path: a step whose context KV (cached +
        # new over its context requests) is at most B = 2**19 rows gathers it into rows [0, N)
        # of one bf16 [B, 576] buffer, out = buf[:N]; md.block_table is [16, 32772].
        Cell(
            why="1M tp4 workspace at N = B = 2**19: a 32768-row chunk behind a 491520-token "
            "prefix, owner row 62, its 16384 pages anywhere in a 16500-block pool (20.3 GiB, "
            "byte offsets past 2**34); every N 1..B a bitwise prefix, rows past N untouched",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_width=32772,
                prefixes=True,
            ),
        ),
        Cell(
            why="1M tp4 workspace at N = B, NoPE source: the nine sharers re-gather layer 35's "
            "row 35 (zero rope) into the same buffer, 491520 cached + 32768 new",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=35,
                out_rows=524288,
                table_width=32772,
                nope=True,
            ),
        ),
        Cell(
            why="1M tp4 packed chunk step: four requests cached 262144/131072/0/2048 + "
            "16384/8192/4096/4096 (N = 428032 of the 2**19-row buffer), owner row 1",
            spec=dict(
                cached=[262144, 131072, 0, 2048],
                new=[16384, 8192, 4096, 4096],
                blocks=13500,
                owners=63,
                tokens_per_block=32,
                row=1,
                out_rows=524288,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tp4 short step in the stale 2**19-row buffer: three requests, prefixes "
            "mid-page (N = 9580), the 32772-column table",
            spec=dict(
                cached=[0, 2047, 5185],
                new=[300, 1, 2047],
                blocks=600,
                owners=63,
                tokens_per_block=32,
                row=7,
                out_rows=524288,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tp4 sixteen requests filling the 2**19-row buffer: 30720 cached + 2048 new "
            "each, kv_req 0..15 over all 16 rows of the 32772-column table",
            spec=dict(
                cached=[30720] * 16,
                new=[2048] * 16,
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=0,
                out_rows=524288,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tp4 boot-sized pool, large: 74183 blocks (91.4 GiB), the 491520 + 32768 "
            "request's 16384 pages the pool's top ones (byte offsets to 98.1e9), N = B",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=74183,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_width=32772,
                far=True,
                large=True,
            ),
        ),
        # A 1M prompt's chunk steps on the workspace path: chunk k is 32768 new rows behind a
        # 32768 * (k - 1)-token prefix, N = 32768 * k, for k = 1..15 (chunk 16 is the N = B
        # cell above; past it the decode path). Every owner layer gathers each chunk at its
        # own row; the chunks here take rows 2, 6, .., 58.
        *(
            Cell(
                why=f"1M tp4 chunk {k} of a 1M prompt at N = {32768 * k}: 32768 new rows behind "
                f"a {32768 * (k - 1)}-token prefix into the 2**19-row buffer, owner row "
                f"{4 * k - 2}, its pages anywhere in a 16500-block pool",
                spec=dict(
                    cached=[32768 * (k - 1)],
                    new=[32768],
                    blocks=16500,
                    owners=63,
                    tokens_per_block=32,
                    row=4 * k - 2,
                    out_rows=524288,
                    table_width=32772,
                ),
            )
            for k in range(1, 16)
        ),
        # A fresh request's dense step: the engine's warmup steps (1 and 2 rows) and a short
        # request (7 rows), into the first rows of the held buffer.
        *(
            Cell(
                why=f"1M tp4 fresh {n}-row step (the engine warmup steps, a short request): "
                f"N = {n} of the stale 2**19-row buffer, its page the pool's last one (the "
                "16500-block pool's top block, owner row 62)",
                spec=dict(
                    cached=[0],
                    new=[n],
                    blocks=16500,
                    owners=63,
                    tokens_per_block=32,
                    row=62,
                    out_rows=524288,
                    table_width=32772,
                    far=True,
                ),
            )
            for n in (1, 2, 7)
        ),
        # The Mistral Large 4 NVFP4 tep4 target's workspace path at max_seq_len 1048576 with
        # chunked prefill: a step whose context KV (cached + new over its context requests) is
        # at most B = 2**19 rows gathers it into rows [0, N) of one bf16 [B, 576] buffer,
        # out = buf[:N], through md.block_table [16, 32772]; owners 63, 32 tokens per block.
        Cell(
            why="1M tep4 workspace at N = B = 2**19: a 32768-row chunk behind a 491520-token "
            "prefix, owner row 62, its 16384 pages anywhere in a 16500-block pool (20.3 GiB, "
            "byte offsets to 2.18e10); every N below B a bitwise prefix of the full call",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                prefixes=True,
            ),
        ),
        Cell(
            why="1M tep4 NoPE source at N = B: the nine sharers re-gather layer 35's row 35 "
            "(zero rope) into the same 2**19-row buffer, 491520 cached + 32768 new",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=35,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                nope=True,
            ),
        ),
        Cell(
            why="1M tep4 packed step, N = 428032: four requests cached 262144/131072/0/2048 "
            "+ new 16384/8192/4096/4096 over a 13500-block pool, owner row 1",
            spec=dict(
                cached=[262144, 131072, 0, 2048],
                new=[16384, 8192, 4096, 4096],
                blocks=13500,
                owners=63,
                tokens_per_block=32,
                row=1,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tep4 short step, N = 9580, leading the 2**19-row buffer: three requests "
            "with prefixes mid-page, owner row 7",
            spec=dict(
                cached=[0, 2047, 5185],
                new=[300, 1, 2047],
                blocks=600,
                owners=63,
                tokens_per_block=32,
                row=7,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tep4 sixteen requests at N = B: 30720 cached + 2048 new each (a full "
            "32768-row step), kv_req 0..15 over every row of the table, owner row 0",
            spec=dict(
                cached=[30720] * 16,
                new=[2048] * 16,
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=0,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tep4 boot-sized pool, large: 74183 blocks (91.4 GiB), the 491520 + 32768 "
            "request's 16384 pages its top ones (byte offsets to 9.81e10), N = B, row 62",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=74183,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                far=True,
                large=True,
            ),
        ),
        # The Mistral Large 4 NVFP4 dep4 target's workspace path at max_seq_len 1048576 with
        # chunked prefill, under attention DP (a rank's own context requests, up to 16): a step
        # whose context KV on the rank (cached + new over its context requests) is at most
        # B = 2**19 rows gathers it into rows [0, N) of one bf16 [B, 576] buffer, out = buf[:N],
        # through md.block_table [16, 32772]; owners 63, 32 tokens per block.
        Cell(
            why="1M dep4 workspace at N = B = 2**19: a 32768-row chunk behind a 491520-token "
            "prefix, owner row 62, its 16384 pages anywhere in a 16500-block pool (20.3 GiB, "
            "byte offsets to 2.18e10); every N below B a bitwise prefix of the full call",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                prefixes=True,
            ),
        ),
        Cell(
            why="1M dep4 NoPE source at N = B: the nine sharers re-gather layer 35's row 35 "
            "(zero rope) into the same 2**19-row buffer, 491520 cached + 32768 new",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=35,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                nope=True,
            ),
        ),
        Cell(
            why="1M dep4 packed step, N = 428032: four of the rank's requests cached "
            "262144/131072/0/2048 + new 16384/8192/4096/4096, a 13500-block pool, owner row 1",
            spec=dict(
                cached=[262144, 131072, 0, 2048],
                new=[16384, 8192, 4096, 4096],
                blocks=13500,
                owners=63,
                tokens_per_block=32,
                row=1,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M dep4 short step, N = 9580, leading the 2**19-row buffer: three requests "
            "with prefixes mid-page, owner row 7",
            spec=dict(
                cached=[0, 2047, 5185],
                new=[300, 1, 2047],
                blocks=600,
                owners=63,
                tokens_per_block=32,
                row=7,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M dep4 sixteen requests at N = B: a rank's 16 own requests, 30720 cached + "
            "2048 new each (a full 32768-row step), kv_req 0..15 over every table row, owner row 0",
            spec=dict(
                cached=[30720] * 16,
                new=[2048] * 16,
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=0,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M dep4 boot-sized pool, large: 74183 blocks (91.4 GiB), the 491520 + 32768 "
            "request's 16384 pages its top ones (byte offsets to 9.81e10), N = B, row 62",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=74183,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                far=True,
                large=True,
            ),
        ),
        # The Mistral Large 4 NVFP4 tep8 target's workspace path at max_seq_len 1048576 with
        # chunked prefill: a step whose context KV (cached + new over its context requests) is
        # at most B = 2**19 rows gathers it into rows [0, N) of one bf16 [B, 576] buffer,
        # out = buf[:N], through md.block_table [16, 32772]; owners 63, 32 tokens per block.
        Cell(
            why="1M tep8 workspace at N = B = 2**19: a 32768-row chunk behind a 491520-token "
            "prefix, owner row 62, its 16384 pages anywhere in a 16500-block pool (20.3 GiB, "
            "byte offsets to 2.18e10); every N below B a bitwise prefix of the full call",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                prefixes=True,
            ),
        ),
        Cell(
            why="1M tep8 NoPE source at N = B: the nine sharers re-gather layer 35's row 35 "
            "(zero rope) into the same 2**19-row buffer, 491520 cached + 32768 new",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=35,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                nope=True,
            ),
        ),
        Cell(
            why="1M tep8 packed step, N = 428032: four requests cached 262144/131072/0/2048 "
            "+ new 16384/8192/4096/4096 over a 13500-block pool, owner row 1",
            spec=dict(
                cached=[262144, 131072, 0, 2048],
                new=[16384, 8192, 4096, 4096],
                blocks=13500,
                owners=63,
                tokens_per_block=32,
                row=1,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tep8 short step, N = 9580, leading the 2**19-row buffer: three requests "
            "with prefixes mid-page, owner row 7",
            spec=dict(
                cached=[0, 2047, 5185],
                new=[300, 1, 2047],
                blocks=600,
                owners=63,
                tokens_per_block=32,
                row=7,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tep8 sixteen requests at N = B: 30720 cached + 2048 new each (a full "
            "32768-row step), kv_req 0..15 over every row of the table, owner row 0",
            spec=dict(
                cached=[30720] * 16,
                new=[2048] * 16,
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=0,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
            ),
        ),
        Cell(
            why="1M tep8 bound-sized pool, large: 147456 blocks (181.6 GiB), the 491520 + 32768 "
            "request's 16384 pages its top ones (byte offsets to 1.95e11), N = B, row 62",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=147456,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_rows=16,
                table_width=32772,
                far=True,
                large=True,
            ),
        ),
    )

    note = """
    Call sites: mistral_large4 targets' context rows, once per owner layer and once
    per step for the NoPE source (its sharers reuse that workspace). Bitwise equal to
    `reference` (the cells). Launch: Triton, grid (N, 5); not captured (context steps).
    A call writes 37.7 MB per 32768 rows and 604 MB at 2**19 rows; the 147456-block cell
    peaks at 185.65 GiB allocated.

    The "1M tp4", "1M tep4", "1M dep4" and "1M tep8" cells are the Mistral Large 4 NVFP4 tp4,
    tep4, dep4 and tep8 targets' workspace path at max_seq_len 1048576 with chunked
    prefill (steps of up to 32768 new rows and 16 requests; under dep4's attention DP, a
    rank's own requests): a step whose context KV (on the rank), cached + new summed over
    its context requests, is at most B = 2**19 rows gathers it into rows [0, N) of one
    bf16 [B, 576] buffer (576 MiB), out = buf[:N], its later rows stale; the owner layers
    at their own row and the NoPE sharers at row 35 again, into the same buffer. A step
    above B gathers nothing (the sparse decode kernel attends it off the pool), so no cell
    passes N > 2**19. Each step a 1M prompt takes on this path is a "1M tp4" cell: its
    chunks k = 1..16, 32768 new rows behind 32768 (k - 1) cached, N = 32768 k (k = 16 the
    N = B cell), and the fresh dense steps of 1, 2 and 7 rows (the engine's warmup steps
    and a short request). The kernel writes rows [0, N) of `out` and nothing past them:
    the cells hold the buffer's later rows to a sentinel, and the N = B cells hold every N
    below B to the full call's leading rows (bitwise). Pages a cell's requests do not
    hold are filled with 0xFF (NaN in every field), so a read from a wrong page, owner
    row or offset changes the bits the cells compare. The tp4, tep4 and dep4 large cells'
    74183-block pool lies above the dep4 boot's fp8_ds_mla pool (64292 blocks per rank,
    measured at fraction 0.9: 2057344 tokens), with the request's pages at its top. The
    tep8 pool at the 1M boot is 114021 blocks per rank (3,648,672 tokens, read back from
    the 1M tep8 boot on fp8_ds_mla at fraction 0.9) and cannot reach 147456 blocks
    (147456 x 32 x 47,136 B = 222.4 GB, more than a rank's whole free memory): the tep8
    large cell's pool is 147456 blocks, the request's pages its top ones.

    Hard limits (sm_103)
      pool byte offset  ((block * owners + row) * tpb + pos % tpb) * 656, int64 in the
                        kernel; past 2**31 bytes from block 1623 row 51, every row from
                        block 1624 on (owners 63, tpb 32); probed at 2.18e10 bytes (16500
                        blocks), 9.81e10 bytes (74183 blocks) and 1.95e11 bytes (147456
                        blocks), the pages the pool's top ones in the large cells  passed
      out row offset    w * out.stride(0), int64; the int32 boundary 2**31 / 576 =
                        3728270 rows lies past every cell (N <= 2**19)      not reached
      kv_req, kv_pos    int32 tensors, widened to int64 in the kernel; kv_req probed to
                        15, kv_pos to 524287                                passed
      block_table       req * block_table.stride(0) + pos // tpb, int64; a [16, 32772]
                        int32 table read to row 15 and column 16383         passed
      N                 grid x; every N 1..64, 2**k - 1 / 2**k / 2**k + 1 from 2**7, every
                        4093rd and every 4099th, every multiple of 32768 and 2**19 - 1,
                        each a prefix of the 2**19 call                     passed
      no index passes through fp32.

    is_valid refuses an `out` with fewer rows than `kv_req` (a shape check, no sync): the
    kernel writes a row for every kv_req entry, past out's end, silently (the guard tests
    show it). It reads kv_pos at every kv_req row as well; nothing compares kv_pos's
    length with kv_req's.
    """

    def __call__(
        self,
        pool: torch.Tensor,
        row: int,
        kv_req: torch.Tensor,
        kv_pos: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        n = kv_req.shape[0]
        if n:
            _gather_kernel[(n, 5)](
                pool, pool.shape[1], row, kv_req, kv_pos, block_table, block_table.stride(0),
                out, out.stride(0), TPB=pool.shape[2],
            )  # fmt: skip
        return out

    def reference(
        self,
        pool: torch.Tensor,
        row: int,
        kv_req: torch.Tensor,
        kv_pos: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """A new tensor shaped like `out`."""
        tpb = pool.shape[2]
        p = kv_pos.long()
        block = block_table[kv_req.long(), p // tpb].long()
        rows = pool[block, row, p % tpb]
        codes = rows[:, :DIM_NOPE].contiguous().view(torch.float8_e4m3fn).float()
        scale = rows[:, DIM_NOPE : DIM_NOPE + 16].contiguous().view(torch.float32)
        nope = (codes.view(-1, DIM_NOPE // TILE, TILE) * scale.unsqueeze(-1)).view(-1, DIM_NOPE)
        rope = rows[:, DIM_NOPE + 16 :].contiguous().view(torch.bfloat16)
        res = torch.empty_like(out)
        res[:, :DIM_NOPE] = nope.to(torch.bfloat16)
        res[:, DIM_NOPE:] = rope
        return res

    def is_valid(
        self,
        pool: torch.Tensor,
        row: int,
        kv_req: torch.Tensor,
        kv_pos: torch.Tensor,
        block_table: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        if pool.dtype != torch.uint8 or pool.dim() != 4 or pool.shape[-1] != ROW_BYTES:
            raise ValueError("pool must be uint8 [blocks, owners, tpb, 656]")
        if not pool.is_contiguous() or not 0 <= row < pool.shape[1]:
            raise ValueError("pool must be contiguous and row one of its owner rows")
        if (
            out.dtype != torch.bfloat16
            or out.shape[-1] != DIM_NOPE + DIM_ROPE
            or out.stride(-1) != 1
        ):
            raise ValueError("out must be bf16 [N, 576] with contiguous rows")
        if out.shape[0] < kv_req.shape[0]:
            raise ValueError(
                f"out has {out.shape[0]} rows for {kv_req.shape[0]} kv_req entries: the kernel "
                "writes a row per entry, past out's end"
            )


ds_mla_dequant_gather = DsMlaDequantGather()
