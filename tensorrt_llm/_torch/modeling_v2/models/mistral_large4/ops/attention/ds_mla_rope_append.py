# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rope the query and key tails and append each token's latent to an fp8_ds_mla pool."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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

#: What every `_ML4_1M` cell shares: the tep4 and tp4 deployments at max_seq_len 1048576 -- 16 query
#: heads (the first of the 64-head fused-query scratch), the 63 KV-owner rows of
#: MistralLarge4CacheManager's latent pool, 32-token pages, the 1048576-row `_rope.indexer`
#: table and md.block_table's 32772 columns (the V2 manager's max_blocks_per_seq at 1M).
_ML4_1M = dict(heads=16, owners=63, tokens_per_block=32, max_pos=1048576, table_width=32772)

#: What every `_ML4_1M_DEP4` cell shares: the dep4 deployment at max_seq_len 1048576 -- all 64
#: query heads of a rank's fused-query scratch (attention DP, no padding heads), 63 KV
#: owners, 32-token pages, the 1048576-row `_rope.indexer` table and the V2 manager's
#: 32772-column `md.block_table`.
_ML4_1M_DEP4 = dict(heads=64, owners=63, tokens_per_block=32, max_pos=1048576, table_width=32772)

#: What every `_ML4_1M_TEP8` cell shares: the tep8 deployment at max_seq_len 1048576 -- 8 query
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
        *tuple(
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
        ),
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
    )

    note = """
    Fused MLA RoPE + fp8_ds_mla paged latent append (32- and 64-token pages, 8/16/64 heads,
    RoPE and NoPE owner layers, CUDA graph decode, and 1M chunked prefill).
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
