# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather paged NVFP4 KV rows and transcode them to E4M3 with one per-tensor scale, via the trtllm
Triton op.

The op is `torch.ops.trtllm.nvfp4_kv_to_e4m3`, registered in
`tensorrt_llm/_torch/custom_ops/nvfp4_kv_to_e4m3.py`.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops.nvfp4_kv_to_e4m3  # noqa: F401 — registers torch.ops.trtllm.nvfp4_kv_to_e4m3
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E4M3_MAX = 448.0
_BLOCK = 16


class Nvfp4KvToE4m3(OpWrapper):
    """Transcode the NVFP4 pool rows each request reads into an E4M3 buffer, in one launch.

    For every request `b` and every position `t < min(seq_lens[b], L)`, with
    `page = block_tables[b, t // page_size]`::

        v[i] = E2M1[code_i] * e4m3(sf[i // 16])  # exact in fp32
        out[b, t, 0, i] = e4m3_rne(clamp(v[i] * inv_scale, -448, 448))  # only if 0 <= page < pages

    The pool is DeepSeek-V4.1's 288-byte compressed-KV row read in place as two
    views: `kv_cache` `[pages, page_size, 1, W/2]` uint8 (packed E2M1, even
    element in the low nibble) and `kv_cache_sf` `[pages, page_size, 1, W/16]`
    E4M3 scale bytes; there is no global scale. Positions at or past
    `seq_lens[b]` (or `L`) and rows whose page id is negative or `>= pages` are
    left unwritten.

    Fusion boundary: page lookup, E2M1 decode, E4M3 scale, rescale and E4M3 cast
    are inside the call; choosing `inv_scale` is the caller's. Writes `out` in
    place and returns it.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"compact r={ratio} inv_scale={inv}: whole-context [8, 132*128/r, 1, 512]",
                spec=dict(kind="compact", ratio=ratio, inv_scale=inv),
            )
            for ratio in (2, 1)
            for inv in (1.0, 16.0, 64.0, 128.0, 0.25, 0.8)
        )
        + tuple(
            Cell(
                why=f"engine pool r={ratio} mbs={mbs}: engine page count, ends named",
                spec=dict(kind="engine", ratio=ratio, pages=pages, mbs=mbs, inv_scale=64.0),
            )
            for mbs in (1, 8, 32)
            for ratio, pages in ((2, 453120), (1, 151040))
        )
        + tuple(
            Cell(
                why=f"per-query top-k r={ratio} T={t}: one-row pages, -1 padded",
                spec=dict(kind="per_query", ratio=ratio, T=t, inv_scale=2.0),
            )
            for ratio in (2, 1)
            for t in (1, 6, 192, 2048, 8192)
        )
    )

    note = """
    Cells. The target's compact compressed buffer `[B, 132 * 128 / r, 1, 512]`
    at compress ratio 2 (64-row pages) and 1 (128-row pages): 8 requests with
    lengths 0, 1, page-1, page+1, 700/r, 4097/r, 16398/r and `L`, one
    page-table hole. The engine's pool page counts (453120 x 64 and
    151040 x 128 rows) with pages named at both ends, into the
    `max_batch_size` views 1, 8, 32. The per-query top-k form the target reads
    when it is smaller than the per-request one: the pool viewed as pages of
    one row (`[pages * bk, 1, 1, C]`, a slot number is its own page id), a
    `[T, 512]` table of selected slots with `-1` after each row's valid
    entries, `seq_lens` 512 for every row, `out` `[T, 512, 1, 512]`.

    Lossy by construction: `E2M1 x E4M3` values carry up to 6 significant bits
    and E4M3 keeps 4, so the result rounds by at most `2^-4` relative in
    E4M3's normal range (43% of values inexact on random rows, about 23% on
    DeepSeek-V4.1's real compressed KV). Code `0x8` (-0.0) keeps its sign;
    `6 * 448` saturates to 448; a NaN scale byte gives NaN.

    Metadata consumed: none. DeepSeek-V4.1's target passes
    `compress_block_tables[r][:B]` and `compressed_kv_lens_cuda[r][:B]`.

    Preconditions: every `t < min(seq_lens[b], L)` has
    `t // page_size < max_pages` (the table is read there); the caller picks an
    `inv_scale` > 0 under which the values it needs do not saturate;
    `block_tables` int32/int64 and `seq_lens` int32; `out` contiguous
    float8_e4m3fn with `W` a power of two. The op raises ValueError on a wrong
    dtype, shape, stride, device or a non-positive `inv_scale`. No host
    synchronization and no allocation: graph-capturable. Grid
    `(ceil(L / 8), B)`, eight rows per program, early exit past `seq_lens[b]`.

    Not certified: `W` other than 512; `max_batch_size` other than 1, 8, 32 in
    the engine-pool case (8 in the others); CUDA-graph capture of the per-query
    form specifically (capture is certified on the whole-context form, the same
    kernel).
    """

    def raw_call(
        self,
        kv_cache: torch.Tensor,
        kv_cache_sf: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        inv_scale: float,
    ) -> torch.Tensor:
        torch.ops.trtllm.nvfp4_kv_to_e4m3(
            kv_cache, kv_cache_sf, block_tables, seq_lens, out, inv_scale
        )
        return out

    def reference(
        self,
        kv_cache: torch.Tensor,
        kv_cache_sf: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: torch.Tensor,
        inv_scale: float,
    ) -> torch.Tensor:
        """The whole of `out` after the call: transcoded rows where the op writes, `out`'s
        prior bytes everywhere else.

        An E2M1 lookup table times the E4M3 scale in fp32 (exact: at most 6
        significant bits), times the fp32 `inv_scale` (one rounding, none for a
        power of two), clamped to +-448 and cast by torch to float8_e4m3fn
        (round-to-nearest-even). `out` is not mutated: the expected buffer is a
        copy of it.
        """
        dev = out.device
        num_requests, length = out.shape[0], out.shape[1]
        pages, page_size = kv_cache.shape[0], kv_cache.shape[1]
        t = torch.arange(length, device=dev)
        col = (t // page_size).clamp_max(block_tables.shape[1] - 1)
        page = block_tables[:, col].long()
        lens = seq_lens.long().clamp_max(length)[:, None]
        written = (t[None, :] < lens) & (page >= 0) & (page < pages)
        page = page.clamp(0, pages - 1)[written]
        row = (t % page_size)[None, :].expand(num_requests, length)[written]
        codes = kv_cache[page, row, 0]
        scales = kv_cache_sf[page, row, 0].contiguous().view(torch.float8_e4m3fn).float()
        lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float32, device=dev)
        values = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], -1).flatten(-2)
        values = values * scales.repeat_interleave(_BLOCK, -1)
        y = values * torch.tensor(inv_scale, dtype=torch.float32, device=dev)
        y = y.clamp(-_E4M3_MAX, _E4M3_MAX).to(torch.float8_e4m3fn)
        expected = out.view(torch.uint8).clone()
        expected[:, :, 0][written] = y.view(torch.uint8)
        return expected.view(out.dtype)

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bitwise on every byte of `out`, written or not.

        The products are exact in fp32 and the cast is torch's RNE on both
        sides, so a differing code is a wrong row, scale or nibble order; a
        differing byte outside the written set is a write the op must not make.
        Controls (nibbles swapped, one scale per 32 instead of 16, `inv_scale`
        ignored) each change more than 30% of the codes.
        """
        assert out.dtype == ref.dtype == torch.float8_e4m3fn, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        bad = (out.view(torch.uint8) != ref.view(torch.uint8)).sum().item()
        assert bad == 0, f"{bad} of {out.numel()} bytes differ"


nvfp4_kv_to_e4m3 = Nvfp4KvToE4m3()
