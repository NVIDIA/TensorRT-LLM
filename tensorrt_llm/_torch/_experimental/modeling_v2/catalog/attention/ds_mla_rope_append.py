# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rope the query and key tails and append each token's latent to an fp8_ds_mla pool."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from .._op import Arch, Cell, OpWrapper

#: One fp8_ds_mla latent row: 512 e4m3 codes | 4 fp32 tile scales | 64 bf16 rope values.
DIM_NOPE = 512
DIM_ROPE = 64
TILE = 128
NUM_TILES = DIM_NOPE // TILE
SCALE_OFFSET = DIM_NOPE
ROPE_OFFSET = DIM_NOPE + 4 * NUM_TILES
TOKEN_BYTES = ROPE_OFFSET + 2 * DIM_ROPE
FP8_MAX = 448.0
#: The floor on a tile's amax / 448 before it is rounded up to a power of two.
SCALE_FLOOR = 1e-4

#: What every `_ML4_1M` cell shares: the tep4 and tp4 targets at max_seq_len 1048576 -- 16 query
#: heads (the first of the 64-head fused-query scratch), the 63 KV-owner rows of
#: MistralLarge4CacheManager's latent pool, 32-token pages, the 1048576-row `_rope.indexer`
#: table and md.block_table's 32772 columns (the V2 manager's max_blocks_per_seq at 1M).
_ML4_1M = dict(heads=16, owners=63, tokens_per_block=32, max_pos=1048576, table_width=32772)

#: What every `_ML4_1M_DEP4` cell shares: the dep4 target at max_seq_len 1048576 -- all 64
#: query heads of a rank's fused-query scratch (attention DP, no padding heads), 63 KV
#: owners, 32-token pages, the 1048576-row `_rope.indexer` table and the V2 manager's
#: 32772-column `md.block_table`.
_ML4_1M_DEP4 = dict(heads=64, owners=63, tokens_per_block=32, max_pos=1048576, table_width=32772)

#: What every `_ML4_1M_TEP8` cell shares: the tep8 target at max_seq_len 1048576 -- 8 query
#: heads (the first of its 64-head fused-query scratch), 63 KV owners, 32-token pages, the
#: 1048576-row `_rope.indexer` table and the V2 manager's 32772-column `md.block_table`.
_ML4_1M_TEP8 = dict(heads=8, owners=63, tokens_per_block=32, max_pos=1048576, table_width=32772)


@triton.jit
def _pow2_ceil(x):
    """2^ceil(log2(x)) for a positive normal fp32 x, exact: from the exponent bits."""
    bits = x.to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    exp = exp + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return (exp << 23).to(tl.float32, bitcast=True)


@triton.jit
def _rope_append_kernel(
    latent_ptr,  # bf16 [T, 576]: RMSNorm(c_kv) | k_pe (unroped)
    lat_stride,
    q_ptr,  # bf16 query rows; its rope tail [.., 512:576] per head, roped in place
    q_stride_t,
    q_stride_h,
    num_q_heads,
    pos_ptr,  # int32/int64 [T]
    table_ptr,  # fp32 [positions, 64]: cos of the 32 pairs, then their sin
    req_ptr,  # int32 [T]: the token's request row in block_table
    block_table_ptr,
    bt_stride,
    pool_ptr,  # uint8 [blocks, owners, tokens_per_block, 656]
    owners,
    row,
    TPB: tl.constexpr,
    ROPE: tl.constexpr,
    HAS_Q: tl.constexpr,
    H_BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    part = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    req = tl.load(req_ptr + t).to(tl.int64)
    block = tl.load(block_table_ptr + req * bt_stride + pos // TPB).to(tl.int64)
    base = ((block * owners + row) * TPB + pos % TPB) * 656
    lat = latent_ptr + t.to(tl.int64) * lat_stride
    half = tl.arange(0, 32)
    if part < 4:
        cols = part * 128 + tl.arange(0, 128)
        xt = tl.load(lat + cols).to(tl.float32)
        # IEEE quotients: an approximate amax / 448 can cross a power of two and double the
        # scale. A power-of-two scale divides exactly; |x / scale| <= 448 by construction.
        ratio = tl.maximum(tl.fdiv(tl.max(tl.abs(xt)), 448.0, ieee_rounding=True), 1e-4)
        scale = _pow2_ceil(ratio)
        codes = tl.fdiv(xt, scale, ieee_rounding=True).to(tl.float8e4nv)
        tl.store(pool_ptr + base + cols, codes.to(tl.uint8, bitcast=True))
        sp = (pool_ptr + base + 512 + part * 4).to(tl.pointer_type(tl.float32))
        tl.store(sp, scale)
    elif part == 4:
        k0 = tl.load(lat + 512 + 2 * half).to(tl.float32)
        k1 = tl.load(lat + 513 + 2 * half).to(tl.float32)
        if ROPE:
            kc = tl.load(table_ptr + pos * 64 + half)
            ks = tl.load(table_ptr + pos * 64 + 32 + half)
            y0 = k0 * kc - k1 * ks
            y1 = k0 * ks + k1 * kc
        else:
            y0 = k0
            y1 = k1
        rp = (pool_ptr + base + 528).to(tl.pointer_type(tl.bfloat16))
        tl.store(rp + 2 * half, y0.to(tl.bfloat16))
        tl.store(rp + 2 * half + 1, y1.to(tl.bfloat16))
    else:
        if HAS_Q and ROPE:
            qc = tl.load(table_ptr + pos * 64 + half)
            qs = tl.load(table_ptr + pos * 64 + 32 + half)
            heads = tl.arange(0, H_BLOCK)
            hmask = heads[:, None] < num_q_heads
            qp = (
                q_ptr
                + t.to(tl.int64) * q_stride_t
                + heads[:, None].to(tl.int64) * q_stride_h
                + 512
                + 2 * half[None, :]
            )
            q0 = tl.load(qp, mask=hmask).to(tl.float32)
            q1 = tl.load(qp + 1, mask=hmask).to(tl.float32)
            tl.store(qp, (q0 * qc[None, :] - q1 * qs[None, :]).to(tl.bfloat16), mask=hmask)
            tl.store(qp + 1, (q0 * qs[None, :] + q1 * qc[None, :]).to(tl.bfloat16), mask=hmask)


def _pow2_ceil_torch(ratio: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(ratio)
    return torch.exp2((exponent - (mantissa == 0.5).to(exponent.dtype)).float())


class DsMlaRopeAppend(OpWrapper):
    """Rope `q`'s rope tail in place and write every token's fp8_ds_mla latent row::

        rope(x)[2i:2i+2] = (x0 c_i - x1 s_i, x0 s_i + x1 c_i)  in fp32, rounded to bf16 once
                          (GPT-J interleaved pairs, c_i / s_i = table[pos, i] / [pos, 32 + i])
        scale[j] = 2^ceil(log2(max(max|c_kv[128j:128j+128]| / 448, 1e-4)))   (fp32)
        row      = [e4m3_rn(c_kv / scale) | scale[0..3] | bf16 rope(k_pe)]    (656 B)

    The row recipe is the reference implementation's fp8_ds_mla writer
    (`concat_and_cache_ds_mla`); a power-of-two scale makes the quotient exact and lets
    a reader that holds the scale as E8M0 read it exactly. `rope=False` (a NoPE layer)
    copies the key tail and leaves `q` alone. The token at position p of the request in
    block-table row r goes to `pool[block_table[r, p // tpb], row, p % tpb]`.

    `latent`: bf16 `[T, 576]`, rows contiguous. `q`: bf16 `[T, H, >= 576]` (the absorbed
    query; its `[..., 512:576]` is roped) or None. `table`: fp32 `[>= max pos + 1, 64]`.
    `pool`: uint8 `[blocks, owners, tokens_per_block, 656]`, contiguous. Returns `pool`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: `tokens` rows of `phase` "decode" (one per request, random positions) or
    # "context" (one request's consecutive positions from `cached`), `heads` query
    # heads (0: no query), `rope`, a pool of `blocks` x `owners` pages of
    # `tokens_per_block`, written at owner `row`; `captured` replays a CUDA graph on new
    # data; `amax` "every_bf16" gives the rows' tiles every finite bf16 magnitude.
    # The `_ML4_1M`, `_ML4_1M_DEP4` and `_ML4_1M_TEP8` cells (`max_pos` set) run on the device,
    # over a table of `max_pos` rows and a `[16, table_width]` block table whose request i
    # is row i: phase "context" (one request of `tokens` rows from `cached`, or
    # `sweep_positions` [lo, hi) in calls of `tokens` rows), "packed" (`contexts`, (cached,
    # new) per request) and "mixed" (`contexts`, then one generation row per
    # `decode_positions` entry); a "decode" cell is captured at each of `graph_batches`
    # (default `tokens`) and replayed on new data, with `positions` "tail" (rows drawn from
    # the table's last 64 positions, one at the last) or "boot" (captured on the engine's
    # warm-up batch, replayed with `live` rows ahead of attention DP's padding dummy,
    # `padding` "one_slot": rows of one request at position 0 whose latents are one row
    # repeated). `far` puts every written page at the top of the pool's `blocks`, the
    # step's last position on its last block.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"decode, {t} rows from a CUDA graph, 16 query heads (tep4/tp4)",
                spec=dict(
                    phase="decode",
                    tokens=t,
                    heads=16,
                    rope=True,
                    blocks=256,
                    owners=63,
                    tokens_per_block=32,
                    row=1,
                    captured=True,
                ),
            )
            for t in (1, 16)
        )
        + (
            Cell(
                why="decode, 64 rows, 64 query heads (dep4), the last owner row",
                spec=dict(
                    phase="decode",
                    tokens=64,
                    heads=64,
                    rope=True,
                    blocks=256,
                    owners=63,
                    tokens_per_block=32,
                    row=62,
                    captured=True,
                ),
            ),
            Cell(
                why="NoPE owner (layer 35's row): no rope, query untouched",
                spec=dict(
                    phase="decode",
                    tokens=16,
                    heads=16,
                    rope=False,
                    blocks=256,
                    owners=63,
                    tokens_per_block=32,
                    row=31,
                ),
            ),
            Cell(
                why="context of 32768 rows from position 0 (the unchunked envelope), 16 heads",
                spec=dict(
                    phase="context",
                    tokens=32768,
                    cached=0,
                    heads=16,
                    rope=True,
                    blocks=1100,
                    owners=63,
                    tokens_per_block=32,
                    row=0,
                ),
            ),
            Cell(
                why="context chunk of 2047 rows after a 5185-token cached prefix, mid-page",
                spec=dict(
                    phase="context",
                    tokens=2047,
                    cached=5185,
                    heads=64,
                    rope=True,
                    blocks=400,
                    owners=63,
                    tokens_per_block=32,
                    row=7,
                ),
            ),
            Cell(
                why="every finite bf16 tile amax once: zero, subnormals, the 1e-4 floor, powers "
                "of two, the largest",
                spec=dict(
                    phase="context",
                    tokens=8160,
                    cached=0,
                    heads=0,
                    rope=True,
                    blocks=300,
                    owners=63,
                    tokens_per_block=32,
                    row=5,
                    amax="every_bf16",
                ),
            ),
            Cell(
                why="64-token pages",
                spec=dict(
                    phase="context",
                    tokens=1000,
                    cached=37,
                    heads=16,
                    rope=True,
                    blocks=64,
                    owners=63,
                    tokens_per_block=64,
                    row=2,
                ),
            ),
            # The 32-token cells above at 64-token pages: the same rows, heads, positions and owner
            # rows, half the blocks for the same token capacity.
            *(
                Cell(
                    why=f"64-token pages: decode, {t} rows from a CUDA graph, 16 query heads "
                    "(tep4/tp4)",
                    spec=dict(
                        phase="decode",
                        tokens=t,
                        heads=16,
                        rope=True,
                        blocks=128,
                        owners=63,
                        tokens_per_block=64,
                        row=1,
                        captured=True,
                    ),
                )
                for t in (1, 16)
            ),
            Cell(
                why="64-token pages: decode, 64 rows, 64 query heads (dep4), the last owner row",
                spec=dict(
                    phase="decode",
                    tokens=64,
                    heads=64,
                    rope=True,
                    blocks=128,
                    owners=63,
                    tokens_per_block=64,
                    row=62,
                    captured=True,
                ),
            ),
            Cell(
                why="64-token pages: NoPE owner (layer 35's row): no rope, query untouched",
                spec=dict(
                    phase="decode",
                    tokens=16,
                    heads=16,
                    rope=False,
                    blocks=128,
                    owners=63,
                    tokens_per_block=64,
                    row=31,
                ),
            ),
            Cell(
                why="64-token pages: context of 32768 rows from position 0 (the unchunked envelope), "
                "16 heads",
                spec=dict(
                    phase="context",
                    tokens=32768,
                    cached=0,
                    heads=16,
                    rope=True,
                    blocks=550,
                    owners=63,
                    tokens_per_block=64,
                    row=0,
                ),
            ),
            Cell(
                why="64-token pages: context chunk of 2047 rows after a 5185-token cached prefix, "
                "mid-page",
                spec=dict(
                    phase="context",
                    tokens=2047,
                    cached=5185,
                    heads=64,
                    rope=True,
                    blocks=200,
                    owners=63,
                    tokens_per_block=64,
                    row=7,
                ),
            ),
            Cell(
                why="64-token pages: every finite bf16 tile amax once: zero, subnormals, the 1e-4 "
                "floor, powers of two, the largest",
                spec=dict(
                    phase="context",
                    tokens=8160,
                    cached=0,
                    heads=0,
                    rope=True,
                    blocks=150,
                    owners=63,
                    tokens_per_block=64,
                    row=5,
                    amax="every_bf16",
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"decode, {t} rows from a CUDA graph, 8 query heads (tep8)",
                spec=dict(
                    phase="decode",
                    tokens=t,
                    heads=8,
                    rope=True,
                    blocks=256,
                    owners=63,
                    tokens_per_block=32,
                    row=1,
                    captured=True,
                ),
            )
            for t in (1, 16)
        )
        + (
            Cell(
                why="context of 32768 rows from position 0 (the unchunked envelope), 8 heads (tep8)",
                spec=dict(
                    phase="context",
                    tokens=32768,
                    cached=0,
                    heads=8,
                    rope=True,
                    blocks=1100,
                    owners=63,
                    tokens_per_block=32,
                    row=0,
                ),
            ),
        )
        + (
            Cell(
                why="1M last chunk, RoPE owner row 62: 32768 rows at 1015808..1048575 behind a "
                "1015808-token cached prefix, 16 of the 64-head scratch (PrefillTarget._attention)",
                spec=dict(
                    _ML4_1M,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=1100,
                    row=62,
                ),
            ),
            Cell(
                why="1M last chunk, NoPE owner (layer 35, row 35): no rope, the query untouched, "
                "32768 rows at 1015808..1048575",
                spec=dict(
                    _ML4_1M,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=False,
                    blocks=1100,
                    row=35,
                ),
            ),
            Cell(
                why="1M packed step, 4 chunks cached 1032192 / 524288 / 0 / 2048 (16384 + 8192 + "
                "4096 + 4096 rows), req_idx 0..3 in one call, a 2.9 GB pool written past 2^31 bytes",
                spec=dict(
                    _ML4_1M,
                    phase="packed",
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    rope=True,
                    blocks=2200,
                    row=1,
                ),
            ),
            Cell(
                why="1M mixed step, 3 chunks cached to 1015808 (one mid-page at 490000) and 13 "
                "generation rows at 0..1048575: req_idx 0..15, 32768 rows in one call",
                spec=dict(
                    _ML4_1M,
                    phase="mixed",
                    contexts=((1015808, 16000), (490000, 10000), (0, 6755)),
                    decode_positions=(
                        1048575,
                        0,
                        2047,
                        2048,
                        4499,
                        1048574,
                        1,
                        699,
                        32768,
                        999999,
                        131071,
                        128,
                        1048544,
                    ),
                    rope=True,
                    blocks=2200,
                    row=0,
                ),
            ),
            Cell(
                why="1M decode graphs at every batch 1..16 (captured; the engine pads to 1/2/4/8/16): "
                "positions in the table's last 64 rows, 1048575 in each, replays bitwise equal to "
                "eager (DecodeTarget._attention)",
                spec=dict(
                    _ML4_1M,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=256,
                    row=1,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    positions="tail",
                ),
            ),
            Cell(
                why="1M NoPE decode graph 16 (captured, layer 35's row 35): positions in the "
                "table's last 64 rows, replayed",
                spec=dict(
                    _ML4_1M,
                    phase="decode",
                    tokens=16,
                    rope=False,
                    blocks=256,
                    row=35,
                    captured=True,
                    positions="tail",
                ),
            ),
            Cell(
                why="1M every table row: positions 0..1048575 in 32 chunk calls of 32768 rows, "
                "owner row 2",
                spec=dict(
                    _ML4_1M,
                    phase="context",
                    tokens=32768,
                    rope=True,
                    blocks=1100,
                    row=2,
                    sweep_positions=(0, 1048576),
                ),
            ),
            Cell(
                why="1M far pool, decode graph 16 (captured, large): 107917 blocks (142.7 GB), every "
                "page at the pool's top, owner row 62, byte offsets to 142,719,800,176",
                spec=dict(
                    _ML4_1M,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=107917,
                    row=62,
                    captured=True,
                    positions="tail",
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M boot pool, last chunk (large): 32768 rows behind 1015808 cached on 74183 "
                "blocks (98.1 GB), every page at the pool's top, byte offsets to 98,106,720,112",
                spec=dict(
                    _ML4_1M,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=74183,
                    row=62,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M last chunk, RoPE owner (row 62): 32768 rows at 1015808..1048575 behind a "
                "1015808-token cached prefix, the first 16 heads of the 64-head scratch "
                "(PrefillTarget._attention)",
                spec=dict(
                    _ML4_1M,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=1100,
                    row=62,
                ),
            ),
            Cell(
                why="1M last chunk, NoPE owner (layer 35, row 35): rope False, 32768 rows at "
                "1015808..1048575, the query untouched",
                spec=dict(
                    _ML4_1M,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=False,
                    blocks=1100,
                    row=35,
                ),
            ),
            Cell(
                why="1M packed step of 4 chunks cached 1032192 / 524288 / 0 / 2048 (16384 + 8192 + "
                "4096 + 4096 rows), req_idx 0..3 in one call, rows past 2^31 bytes of a 2200-block "
                "pool",
                spec=dict(
                    _ML4_1M,
                    phase="packed",
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    rope=True,
                    blocks=2200,
                    row=1,
                ),
            ),
            Cell(
                why="1M packed step of 16 chunk tails, 2048 rows each behind cached prefixes "
                "1046528 - 65537 i (15 of them mid-page), req_idx 0..15 in one call",
                spec=dict(
                    _ML4_1M,
                    phase="packed",
                    contexts=tuple((1046528 - 65537 * i, 2048) for i in range(16)),
                    rope=True,
                    blocks=1100,
                    row=40,
                ),
            ),
            Cell(
                why="1M mixed step: 3 chunks cached to 1015808 (one from mid-page 490000), then 13 "
                "generation rows at 0..1048575, req_idx 0..15, 32768 rows in one call",
                spec=dict(
                    _ML4_1M,
                    phase="mixed",
                    contexts=((1015808, 16000), (490000, 10000), (0, 6755)),
                    decode_positions=(
                        1048575,
                        0,
                        2047,
                        2048,
                        4499,
                        1048574,
                        1,
                        699,
                        32768,
                        999999,
                        131071,
                        128,
                        1048544,
                    ),
                    rope=True,
                    blocks=2200,
                    row=0,
                ),
            ),
            Cell(
                why="1M decode graphs at every batch 1..16 (captured; the engine pads to "
                "1/2/4/8/16): positions in the table's last 64 rows, 1048575 in each, every replay "
                "bitwise equal to eager (DecodeTarget._attention)",
                spec=dict(
                    _ML4_1M,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=256,
                    row=1,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    positions="tail",
                ),
            ),
            Cell(
                why="1M NoPE decode graph 16 (captured, layer 35's row 35): positions in the "
                "table's last 64 rows, the query untouched",
                spec=dict(
                    _ML4_1M,
                    phase="decode",
                    tokens=16,
                    rope=False,
                    blocks=256,
                    row=35,
                    captured=True,
                    positions="tail",
                ),
            ),
            Cell(
                why="1M boot-scale pool, decode graph 16 (captured, large): 74183 blocks (98.1 GB), "
                "every page at the pool's top, its last row written at byte 98,106,720,112",
                spec=dict(
                    _ML4_1M,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=74183,
                    row=62,
                    captured=True,
                    positions="tail",
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="1M boot-scale pool, last chunk (large): 32768 rows at 1015808..1048575 on "
                "74183 blocks (98.1 GB), every page at the pool's top, its last row written",
                spec=dict(
                    _ML4_1M,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=74183,
                    row=62,
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="dep4 1M last chunk, RoPE owner (row 62): 32768 rows at 1015808..1048575 behind "
                "a 1015808-token cached prefix, all 64 query heads roped (PrefillTarget._attention)",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=1100,
                    row=62,
                ),
            ),
            Cell(
                why="dep4 1M last chunk, NoPE owner (layer 35, row 35): rope False, 32768 rows at "
                "1015808..1048575, the 64-head query untouched",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=False,
                    blocks=1100,
                    row=35,
                ),
            ),
            Cell(
                why="dep4 1M packed step of 4 chunks cached 1032192 / 524288 / 0 / 2048 (16384 + "
                "8192 + 4096 + 4096 rows), req_idx 0..3 in one call, rows past 2^31 bytes of a "
                "2200-block pool",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="packed",
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    rope=True,
                    blocks=2200,
                    row=1,
                ),
            ),
            Cell(
                why="dep4 1M packed step of 16 chunk tails, 2048 rows each behind cached prefixes "
                "1046528 - 65537 i (15 of them mid-page), req_idx 0..15 in one call",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="packed",
                    contexts=tuple((1046528 - 65537 * i, 2048) for i in range(16)),
                    rope=True,
                    blocks=1100,
                    row=40,
                ),
            ),
            Cell(
                why="dep4 1M mixed step: 3 chunks cached to 1015808 (one from mid-page 490000), then "
                "13 generation rows at 0..1048575, req_idx 0..15, 32768 rows in one call",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="mixed",
                    contexts=((1015808, 16000), (490000, 10000), (0, 6755)),
                    decode_positions=(
                        1048575,
                        0,
                        2047,
                        2048,
                        4499,
                        1048574,
                        1,
                        699,
                        32768,
                        999999,
                        131071,
                        128,
                        1048544,
                    ),
                    rope=True,
                    blocks=2200,
                    row=0,
                ),
            ),
            Cell(
                why="dep4 1M decode graphs at every batch 1..16 (captured; the engine pads to "
                "1/2/4/8/16, and runs any batch eagerly beside a prefilling peer): positions in the "
                "table's last 64 rows, 1048575 in each, every replay bitwise equal to eager "
                "(DecodeTarget._attention)",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=256,
                    row=1,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    positions="tail",
                ),
            ),
            Cell(
                why="dep4 1M boot captures 1/2/4/8/16 under attention DP's padding: captured on the "
                "warm-up batch (a key at 1048574 ahead of one-key fillers at 0, the warm-up call "
                "gated), replayed with one live row at 1048575, at 0 (an idle rank's dummy) and at a "
                "drawn position ahead of the padding dummy repeated on one slot; every replay "
                "bitwise the eager call and the call with the padding cut to one row",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=256,
                    row=1,
                    captured=True,
                    graph_batches=(1, 2, 4, 8, 16),
                    live=1,
                    padding="one_slot",
                    positions="boot",
                ),
            ),
            Cell(
                why="dep4 1M boot captures at owner row 0, Triton's multiple-of-16 class of `row` "
                "(DecodeTarget's rows 0/16/32/48, layers 0/16/32/52): graphs 1/2/4/8/16 captured on "
                "the warm-up batch and replayed as the row-1 boot cell replays them",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=256,
                    row=0,
                    captured=True,
                    graph_batches=(1, 2, 4, 8, 16),
                    live=1,
                    padding="one_slot",
                    positions="boot",
                ),
            ),
            Cell(
                why="dep4 1M NoPE decode graph 16 (captured, layer 35's row 35): positions in the "
                "table's last 64 rows, the 64-head query untouched",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="decode",
                    tokens=16,
                    rope=False,
                    blocks=256,
                    row=35,
                    captured=True,
                    positions="tail",
                ),
            ),
            Cell(
                why="dep4 1M every table row: positions 0..1048575 in 32 chunk calls of 32768 rows, "
                "all 64 heads, owner row 2",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="context",
                    tokens=32768,
                    rope=True,
                    blocks=1100,
                    row=2,
                    sweep_positions=(0, 1048576),
                ),
            ),
            Cell(
                why="dep4 1M far pool, decode graph 16 (captured, large): 107917 blocks (142.7 GB), "
                "every page at the pool's top, its last row written at byte 142,719,800,176",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=107917,
                    row=62,
                    captured=True,
                    positions="tail",
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="dep4 1M boot-scale pool, last chunk (large): 32768 rows at 1015808..1048575 on "
                "74183 blocks (98.1 GB), every page at the pool's top, its last row written at byte "
                "98,106,720,112",
                spec=dict(
                    _ML4_1M_DEP4,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=74183,
                    row=62,
                    far=True,
                    large=True,
                ),
            ),
        )
        + (
            Cell(
                why="tep8 1M last chunk, RoPE owner (row 62): 32768 rows at 1015808..1048575 "
                "behind a 1015808-token cached prefix, the first 8 heads of the 64-head scratch "
                "(PrefillTarget._attention)",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=1100,
                    row=62,
                ),
            ),
            Cell(
                why="tep8 1M last chunk, NoPE owner (layer 35, row 35): rope False, 32768 rows at "
                "1015808..1048575, the query untouched",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=False,
                    blocks=1100,
                    row=35,
                ),
            ),
            Cell(
                why="tep8 1M packed step of 4 chunks cached 1032192 / 524288 / 0 / 2048 (16384 + "
                "8192 + 4096 + 4096 rows), req_idx 0..3 in one call, rows past 2^31 bytes of a "
                "2200-block pool",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="packed",
                    contexts=((1032192, 16384), (524288, 8192), (0, 4096), (2048, 4096)),
                    rope=True,
                    blocks=2200,
                    row=1,
                ),
            ),
            Cell(
                why="tep8 1M packed step of 16 chunk tails, 2048 rows each behind cached prefixes "
                "1046528 - 65537 i (15 of them mid-page), req_idx 0..15 in one call",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="packed",
                    contexts=tuple((1046528 - 65537 * i, 2048) for i in range(16)),
                    rope=True,
                    blocks=1100,
                    row=40,
                ),
            ),
            Cell(
                why="tep8 1M mixed step: 3 chunks cached to 1015808 (one from mid-page 490000), "
                "then 13 generation rows at 0..1048575, req_idx 0..15, 32768 rows in one call",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="mixed",
                    contexts=((1015808, 16000), (490000, 10000), (0, 6755)),
                    decode_positions=(
                        1048575,
                        0,
                        2047,
                        2048,
                        4499,
                        1048574,
                        1,
                        699,
                        32768,
                        999999,
                        131071,
                        128,
                        1048544,
                    ),
                    rope=True,
                    blocks=2200,
                    row=0,
                ),
            ),
            Cell(
                why="tep8 1M decode graphs at every batch 1..16 (captured; the engine pads to "
                "1/2/4/8/16): positions in the table's last 64 rows, 1048575 in each, every replay "
                "bitwise equal to eager (DecodeTarget._attention)",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=256,
                    row=1,
                    captured=True,
                    graph_batches=tuple(range(1, 17)),
                    positions="tail",
                ),
            ),
            Cell(
                why="tep8 1M boot captures at owner row 0, Triton's multiple-of-16 class of `row` "
                "(DecodeTarget._attention's rows 0/16/32/48, layers 0/16/32/52): decode graphs at "
                "1/2/4/8/16, positions in the table's last 64 rows, every replay bitwise equal to "
                "eager",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=256,
                    row=0,
                    captured=True,
                    graph_batches=(1, 2, 4, 8, 16),
                    positions="tail",
                ),
            ),
            Cell(
                why="tep8 1M NoPE decode graph 16 (captured, layer 35's row 35): positions in the "
                "table's last 64 rows, the query untouched",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="decode",
                    tokens=16,
                    rope=False,
                    blocks=256,
                    row=35,
                    captured=True,
                    positions="tail",
                ),
            ),
            Cell(
                why="tep8 1M every table row: positions 0..1048575 in 32 chunk calls of 32768 "
                "rows, owner row 2",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="context",
                    tokens=32768,
                    rope=True,
                    blocks=1100,
                    row=2,
                    sweep_positions=(0, 1048576),
                ),
            ),
            Cell(
                why="tep8 1M pool bound, decode graph 16 (captured, large): 147456 blocks (195.0 "
                "GB), every page at the pool's top, its last row written at byte 195,009,969,520",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="decode",
                    tokens=16,
                    rope=True,
                    blocks=147456,
                    row=62,
                    captured=True,
                    positions="tail",
                    far=True,
                    large=True,
                ),
            ),
            Cell(
                why="tep8 1M pool bound, last chunk (large): 32768 rows at 1015808..1048575 on "
                "147456 blocks (195.0 GB), every page at the pool's top, its last row written",
                spec=dict(
                    _ML4_1M_TEP8,
                    phase="context",
                    tokens=32768,
                    cached=1015808,
                    rope=True,
                    blocks=147456,
                    row=62,
                    far=True,
                    large=True,
                ),
            ),
        )
    )

    note = """
    Call sites: every mistral_large4 target's owner layers, context and generation
    rows in one call, before the layer's attention reads the pool. A sharing layer
    (NoPE) writes nothing and its query tail is zero.

    The Mistral Large 4 NVFP4 tp4 target at max_seq_len 1048576 (chunked prefill,
    max_num_tokens 32768, max_batch_size 16, decode captured at 1/2/4/8/16 padded):
    PrefillTarget._attention (eager) and DecodeTarget._attention (captured) on the 63 KV
    owners, one call per owner layer over every row of the step. `q` is the first 16
    heads of the step's [T, 64, 576] fused-query scratch (heads 16..63 are never
    touched), `table` the target's `_rope.indexer` (fp32 [1048576, 64], cos | sin rounded
    through bf16), `pos` the step's position_ids and `req_idx` md.req_idx_per_token (both
    int32; a generation row's request row is num_contexts + its index), `block_table`
    md.block_table [16, 32772] int32, `pool` the manager's get_latent_pool() at `row` =
    latent_pool_row(layer), and `rope` False on the NoPE owners. The `_ML4_1M` cells cover
    that envelope: a 32768-row chunk at 1015808..1048575 on a RoPE and on the NoPE
    owner, every position 0..1048575 in 32 chunk calls, a packed step of 4 chunks, a
    mixed step of 3 chunks and 13 generation rows (16 requests: block-table rows 0..15,
    column 32767 of row 15), decode graphs captured at every batch 1..16 with a row at
    1048575 (each replay bitwise equal to an eager call on the same inputs), and two
    pools at the far end (large): 74183 blocks, the delivered 1M tp4 boot's (this pool's
    own 1M tp4 boot measured 72757 blocks, 2328224 tokens), and 107917, each with the
    pool's last slot written. Cost on sm_103: 0.02 to 6.7 s per pre-merge cell, most of
    it Triton's first compile of a specialization (the 32-call sweep, compiled already,
    1.6 s); the large cells 1.2 s and 0.2 s.

    The Mistral Large 4 NVFP4 tep4 target at max_seq_len 1048576 (chunked prefill,
    max_num_tokens 32768, max_batch_size 16, decode captured at 1/2/4/8/16 padded):
    PrefillTarget._attention (eager) and DecodeTarget._attention (captured) on the 63 KV
    owners, one call per owner layer over every row of the step, a step's context chunk
    rows first and its generation rows after them. `latent` is `[RMSNorm(c_kv) | k_pe]`
    (k_pe zero on a NoPE owner); `q` the first 16 heads of the step's `[T, 64, 576]`
    fused-query scratch, whose heads 16..63 the call leaves as they are; `table` the
    target's `_rope.indexer` (fp32 `[1048576, 64]`, cos | sin rounded through bf16);
    `pos` the position_ids and `req_idx` md.req_idx_per_token (both int32; a generation
    row's request is num_contexts + its index); `block_table` md.block_table (int32
    `[16, 32772]`, the V2 manager's max_blocks_per_seq at 1M); `pool` the manager's
    get_latent_pool() at `row` = latent_pool_row(layer); `rope` False on the NoPE owners
    (layer 35 at row 35 and the `i % 4 == 3` owners below it). The `_ML4_1M` cells cover
    that envelope: a 32768-row chunk at 1015808..1048575 on a RoPE owner and on the NoPE
    owner; every position 0..1048575 in 32 chunk calls; packed steps of 4 chunks and of
    16 chunk tails; a mixed step of 3 chunks and 13 generation rows (16 requests, block
    table rows 0..15, column 32767 of rows 3, 8 and 15); decode graphs captured at every
    batch 1..16 with a row at 1048575, each replay bitwise equal to an eager call on the
    same inputs; and, large, a pool of 74183 blocks with its last row written, by a
    decode graph and by a last chunk. 74183 is above the tep4 target's 1M boot pool (72757
    blocks per rank, 2328224 tokens, as the target's boot measured it on this
    pool). Cost on sm_103: 0.03 to 1.7 s per pre-merge `_ML4_1M` cell (the first
    one builds the 1048576-row table; the 32-call sweep 1.7 s), pools to 2.9 GB; the two
    large cells 1.4 s together on a 98.1 GB pool.

    The Mistral Large 4 NVFP4 dep4 target at max_seq_len 1048576 (attention DP over 4
    ranks, each running all 64 heads on its own requests; chunked prefill,
    max_num_tokens 32768 and max_batch_size 16 per rank, decode captured at
    1/2/4/8/16 padded): PrefillTarget._attention (eager) and DecodeTarget._attention
    (captured; eager on a rank whose peer prefills) on the 63 KV owners, one call per
    owner layer over every row of the rank's step, its context chunk rows first and
    its generation rows after them. `latent` is `[RMSNorm(c_kv) | k_pe]` (k_pe zero on
    a NoPE owner); `q` the whole `[T, 64, 576]` fused-query scratch, every head
    roped; `table` the target's `_rope.indexer` (fp32 `[1048576, 64]`, cos | sin
    rounded through bf16); `pos` the position_ids and `req_idx`
    md.req_idx_per_token (both int32; a generation row's request is num_contexts +
    its index); `block_table` md.block_table (int32 `[16, 32772]`, the V2 manager's
    max_blocks_per_seq at 1M); `pool` the manager's get_latent_pool() at `row` =
    latent_pool_row(layer); `rope` False on the NoPE owners (layer 35 at row 35 and
    the `i % 4 == 3` owners below it). The `_ML4_1M_DEP4` cells cover that envelope:
    a 32768-row chunk at 1015808..1048575 on a RoPE owner and on the NoPE owner;
    every position 0..1048575 in 32 chunk calls; packed steps of 4 chunks and of 16
    chunk tails; a mixed step of 3 chunks and 13 generation rows (16 requests,
    block-table column 32767 of rows 3, 8 and 15); decode graphs captured at every
    batch 1..16 with a row at 1048575, each replay bitwise equal to an eager call on
    the same inputs; the NoPE decode graph at 16; the boot captures under attention
    DP's padding (below), at owner row 1 and again at row 0; and, large, a
    107917-block pool written by a decode graph
    and a 74183-block pool written by a last chunk, each up to its last row. Both
    are above this target's 1M boot pool (64292 blocks per rank, measured by its
    fp8_ds_mla boot at fraction 0.9: 2057344 tokens). Cost
    on sm_103: 0.02 to 1.44 s per pre-merge `_ML4_1M_DEP4` cell (the first one builds
    the 1048576-row table; the 32-call sweep the longest), pools to 2.9 GB; the two
    large cells 1.4 s together on pools of 142.7 and 98.1 GB.

    Attention DP's CUDA-graph padding (dep4): `CUDAGraphRunner._get_padded_batch`
    pads every rank's decode batch to the largest rank's with one padding dummy
    repeated, so a graph replays a rank's live rows ahead of rows of one request at
    position 0, each with its own req_idx and its own block-table row holding the
    dummy's page: they all write one slot. The engine zero-fills a CUDA-graph
    dummy's input id (`_zero_fill_row_spans` in
    `pyexecutor/engine/runners/decoder/inputs.py`), so the target hands this call
    one latent row repeated for them (inferred: the layers before the write compute
    each row on its own). Rows with bitwise-identical latents on one slot leave it
    as one of them alone does: certified at batches 1/2/4/8/16, captured on the
    warm-up batch (the longest dummy's new key at 1048574 ahead of one-key fillers
    at 0, each on a page of its own; the warm-up call gated) and replayed with the
    live row at 1048575, at 0 (an idle rank's dummy) and at a drawn position, each
    replay bitwise the eager call and the eager call with the padding cut to one
    row; probed once more over 500 eager calls and 500 replays at batch 16, every
    one bitwise one row's write. Rows with different latents on one slot race:
    probed over 1000 calls with 15 such rows, the slot held no one row's whole
    write in any of them, its code tiles, scales and rope bytes coming from several
    rows, and its rope bytes were themselves a byte mix of rows in 32.

    The Mistral Large 4 NVFP4 tep8 target at max_seq_len 1048576 (chunked prefill,
    max_num_tokens 32768, max_batch_size 16, decode captured at 1/2/4/8/16 padded):
    PrefillTarget._attention (eager) and DecodeTarget._attention (captured) on the 63 KV
    owners, one call per owner layer over every row of the step, a step's context chunk
    rows first and its generation rows after them. `latent` is `[RMSNorm(c_kv) | k_pe]`
    (k_pe zero on a NoPE owner); `q` the first 8 heads of the step's `[T, 64, 576]`
    fused-query scratch, whose heads 8..63 the call leaves as they are (the kernel's
    16-head block masks heads 8..15); `table` the target's `_rope.indexer` (fp32
    `[1048576, 64]`, cos | sin rounded through bf16); `pos` the position_ids and `req_idx`
    md.req_idx_per_token (both int32; a generation row's request is num_contexts + its
    index); `block_table` md.block_table (int32 `[16, 32772]`, the V2 manager's
    max_blocks_per_seq at 1M); `pool` the manager's get_latent_pool() at `row` =
    latent_pool_row(layer); `rope` False on the NoPE owners (layer 35 at row 35 and the
    `i % 4 == 3` owners below it). The `_ML4_1M_TEP8` cells cover that envelope: a
    32768-row chunk at 1015808..1048575 on a RoPE owner and on the NoPE owner; every
    position 0..1048575 in 32 chunk calls; packed steps of 4 chunks and of 16 chunk tails;
    a mixed step of 3 chunks and 13 generation rows (16 requests, block table rows 0..15,
    column 32767 of rows 3, 8 and 15); decode graphs captured at every batch 1..16 with a
    row at 1048575, each replay bitwise equal to an eager call on the same inputs, at owner
    row 1 and again at row 0 for the captured batches 1/2/4/8/16; and, large, a pool of
    147456 blocks with its last row written, by a decode graph and by a last chunk.
    147456 is the tep8 pool bound, above this target's 1M boot pool (114021 blocks per
    rank, 3,648,672 tokens, read back from its fp8_ds_mla boot at fraction 0.9). Cost on
    sm_103: 0.02 to 1.5 s per pre-merge `_ML4_1M_TEP8` cell (the first one builds the
    1048576-row table; the 32-call sweep 1.5 s), at most 14.6 GiB of device memory,
    pools to 2.9 GB; the two large cells 0.3 s each, peaks of 184.2 and 193.4 GiB on a
    195.0 GB pool.

    Bytes: the cells gate the codes and the scales bitwise against `reference`, the
    rope halves (key and query) within one bf16 ulp (an FMA in the rotation may
    round once less; none did on sm_103 so far), and every other pool byte bitwise.
    With `rope=False` the row is bitwise the reference implementation's writer
    (`concat_and_cache_mla`, fp8_ds_mla) on the same input: 20,480 rows, tile amaxes
    2^-30..2^15 and zero tiles, measured once on sm_103.
    Launch: Triton, grid (T, 6) -- four quantized tiles, the key rope, the query
    rope; CUDA-graph capturable. Triton specializes the int `row` three ways (equal
    to 1, a multiple of 16, any other), so the dep4 and the tep8 DecodeTarget's captured
    calls run three compiled kernels with rope (owner row 1; rows 0, 16, 32 and 48; the
    rest) and one without (every NoPE owner row, 3, 7, ..., 35, is in the third class).
    A captured 64-head cell holds each: the boot captures and decode graphs at row 1,
    the boot captures at row 0, the far decode graph at 62, the NoPE graph at 35. So
    does a captured 8-head cell: the decode graphs at row 1 and at row 0, the far decode
    graph at 62, the NoPE graph at 35.

    Preconditions the op does not check: positions below the table's rows, every
    token's block-table entry allocated, `row < owners`, no two tokens on one slot
    unless their latent rows are bitwise identical (attention DP's padding, above).

    Hard limits (sm_103). The kernel widens pos, req and block to int64 before any
    address arithmetic, so every offset below is int64.
      pool byte offset  ((block * owners + row) * tpb + pos % tpb) * 656: at 63 owners
                        x 32 a row passes 2^31 bytes from slot 3,273,604 (block 1623)
                        and 2^37 from block 103,923; probed at 2200 blocks (rows to
                        byte 2,906,887,536), at 74183 (the last row at byte
                        98,106,720,112), at 107917 (the last row at byte
                        142,719,800,176) and, at tep8's 8 heads, at 147456 (the last
                        row at byte 195,009,969,520)  passed
      table offset      pos * 64 + 63 <= 67,108,863 at position 1048575, 2^31 not
                        reached; probed at every position 0..1048575  passed
      block table       req * 32772 + pos // 32 <= 524,347 (row 15, column 32767),
                        2^31 not reached; probed at 524,347  passed
      q offset          t * 36864 + h * 576 + 575 elements in the 64-head scratch
                        passes 2^31 at 58,255 rows (row 58,254 at 64 heads; at h <= 7,
                        row 58,255), past max_num_tokens 32768 (1,207,959,552 elements
                        there, 56% of 2^31); not reached (rows to 32767 driven at 8, at
                        16 and at all 64 heads)
      latent offset     t * 576 <= 18,874,368 elements at 32768 rows; not reached
      fp32 index        none: positions and slots never pass through fp32
    Not certified: positions at or past the table's rows (the kernel reads past its
    end unchecked), pools past 147456 blocks (past 107917 at 16 and 64 heads), more
    than 32768 rows in one call, query widths other than 0, 8, 16 and 64 heads, pages
    other than 32 at 1M positions, owner counts other than 63 (and 1, in the scale
    test), rows with different latents on one slot (they race, above).
    """

    def __call__(
        self,
        latent: torch.Tensor,
        q: Optional[torch.Tensor],
        pos: torch.Tensor,
        table: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        pool: torch.Tensor,
        row: int,
        rope: bool,
    ) -> torch.Tensor:
        tokens = latent.shape[0]
        if tokens == 0:
            return pool
        heads = 0 if q is None else q.shape[1]
        _rope_append_kernel[(tokens, 6)](
            latent, latent.stride(0),
            latent if q is None else q, 0 if q is None else q.stride(0),
            0 if q is None else q.stride(1), heads,
            pos, table, req_idx, block_table, block_table.stride(0),
            pool, pool.shape[1], row,
            TPB=pool.shape[2], ROPE=bool(rope), HAS_Q=q is not None,
            H_BLOCK=max(16, triton.next_power_of_2(max(heads, 1))),
        )  # fmt: skip
        return pool

    def reference(
        self,
        latent: torch.Tensor,
        q: Optional[torch.Tensor],
        pos: torch.Tensor,
        table: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        pool: torch.Tensor,
        row: int,
        rope: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """`(pool, q)` after the call, as new tensors; the inputs are not written."""
        p = pos.long()
        tpb = pool.shape[2]
        x = latent[:, :DIM_NOPE].float().view(-1, NUM_TILES, TILE)
        ratio = (x.abs().amax(-1) / FP8_MAX).clamp_min(SCALE_FLOOR)
        scale = _pow2_ceil_torch(ratio)
        codes = (x / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

        def rotate(tail):
            pair = tail.float().reshape(*tail.shape[:-1], DIM_ROPE // 2, 2)
            c = table[p, : DIM_ROPE // 2].view(-1, *([1] * (pair.dim() - 3)), DIM_ROPE // 2)
            s = table[p, DIM_ROPE // 2 :].view_as(c)
            out = torch.stack(
                [pair[..., 0] * c - pair[..., 1] * s, pair[..., 0] * s + pair[..., 1] * c], -1
            )
            return out.reshape(tail.shape).to(torch.bfloat16)

        k_pe = latent[:, DIM_NOPE:]
        k_pe = rotate(k_pe) if rope else k_pe.to(torch.bfloat16)
        packed = torch.cat(
            [
                codes.reshape(-1, DIM_NOPE).view(torch.uint8),
                scale.contiguous().view(torch.uint8),
                k_pe.contiguous().view(torch.uint8),
            ],
            dim=1,
        )
        out = pool.clone()
        block = block_table[req_idx.long(), p // tpb].long()
        out[block, row, p % tpb] = packed
        q_out = None
        if q is not None:
            q_out = q.clone()
            if rope:
                q_out[..., DIM_NOPE : DIM_NOPE + DIM_ROPE] = rotate(
                    q[..., DIM_NOPE : DIM_NOPE + DIM_ROPE]
                )
        return out, q_out

    def is_valid(
        self,
        latent: torch.Tensor,
        q: Optional[torch.Tensor],
        pos: torch.Tensor,
        table: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        pool: torch.Tensor,
        row: int,
        rope: bool,
    ) -> None:
        # Each would address the wrong bytes without an error.
        if latent.dtype != torch.bfloat16 or latent.shape[-1] != DIM_NOPE + DIM_ROPE:
            raise ValueError(
                f"latent must be bf16 [T, 576], got {latent.dtype} {tuple(latent.shape)}"
            )
        if latent.stride(1) != 1:
            raise ValueError("latent rows must be contiguous")
        if pool.dtype != torch.uint8 or pool.dim() != 4 or pool.shape[-1] != TOKEN_BYTES:
            raise ValueError(
                f"pool must be uint8 [blocks, owners, tpb, 656], got {tuple(pool.shape)}"
            )
        if not pool.is_contiguous():
            raise ValueError("pool must be contiguous")
        if not 0 <= row < pool.shape[1]:
            raise ValueError(f"row {row} outside the pool's {pool.shape[1]} owner rows")
        if table.dtype != torch.float32 or table.shape[-1] != DIM_ROPE:
            raise ValueError("table must be fp32 [positions, 64]: 32 cos then 32 sin")
        if q is not None and (
            q.dtype != torch.bfloat16 or q.shape[-1] < DIM_NOPE + DIM_ROPE or q.stride(2) != 1
        ):
            raise ValueError("q must be bf16 [T, H, >= 576] with contiguous heads")


ds_mla_rope_append = DsMlaRopeAppend()
