# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather context requests' fp8_ds_mla latent rows out of the paged pool, dequantized to bf16."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

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
            why="one 32768-token request (unchunked envelope)",
            spec=dict(cached=[0], new=[32768], blocks=1100, owners=63, tokens_per_block=32, row=31),
        ),
        Cell(
            why="three requests, cached prefixes mid-page, last owner row",
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
            why="sixteen 2048-token requests (full prefill batch)",
            spec=dict(
                cached=[0] * 16, new=[2048] * 16, blocks=1100, owners=63, tokens_per_block=32, row=0
            ),
        ),
        Cell(
            why="64-token pages with cached prefix",
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
            why="1M workspace at N = B = 2**19 behind cached prefix, far pool, owner row 62",
            spec=dict(
                cached=[491520],
                new=[32768],
                blocks=16500,
                owners=63,
                tokens_per_block=32,
                row=62,
                out_rows=524288,
                table_width=32772,
                far=True,
            ),
        ),
        Cell(
            why="1M workspace NoPE source (owner row 35) and packed chunks in stale buffer",
            spec=dict(
                cached=[262144, 131072, 0, 2048],
                new=[16384, 8192, 4096, 4096],
                blocks=14000,
                owners=63,
                tokens_per_block=32,
                row=35,
                out_rows=524288,
                table_width=32772,
                nope=True,
            ),
        ),
    )

    note = """
    Triton dequantize-gather from uint8 fp8_ds_mla paged pool into a contiguous bf16 [N, 576]
    context workspace. Bitwise exact against reference (e4m3 * power-of-two scale is exact in
    bf16; rope half is copied verbatim).
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
