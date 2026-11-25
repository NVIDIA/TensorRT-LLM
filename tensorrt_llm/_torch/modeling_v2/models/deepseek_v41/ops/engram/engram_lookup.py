# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram gather: rows of an MXFP8 (E4M3) or MXFP4 (packed E2M1) table with UE8M0 scales, dequantized to bf16 in one
pass."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.kernels.blackwell.engram.lookup import compile_engram_lookup
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# E4M3 bytes of the E2M1 codes 0-15 (0, 0.5, 1, 1.5, 2, 3, 4, 6, then the same negated): exact, as the kernel widens.
_E2M1_TO_E4M3 = (
    0x00,
    0x30,
    0x38,
    0x3C,
    0x40,
    0x44,
    0x48,
    0x4C,
    0x80,
    0xB0,
    0xB8,
    0xBC,
    0xC0,
    0xC4,
    0xC8,
    0xCC,
)


def e2m1_to_e4m3_bytes(packed: torch.Tensor) -> torch.Tensor:
    """uint8 `[..., n]` of E2M1 pairs (element 2k in byte k's low nibble) -> uint8 `[..., 2n]` E4M3 bytes."""
    lut = torch.tensor(_E2M1_TO_E4M3, dtype=torch.uint8, device=packed.device)
    nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).long()
    return lut[nibbles]


class EngramLookup(OpWrapper):
    """`out [T, local_heads, D]`: the dequantized table rows of every token's hash ids.

    Row of (token t, head h) = `ids[t, head_start + h] + offsets[h]`. With `masked`, a row outside
    `[vocab_start, vocab_end)` belongs to another shard: it reads as zero, and an owned row is addressed
    relative to `vocab_start`. Without it the row is the table row itself (head shards rebase `offsets`).
    The table may be device memory or a UVA view of pinned host memory. `out` is written and returned. With `mxfp4` the
    table rows are `D / 2` bytes of E2M1 pairs (stock's MXFP4 Engram tables), widened exactly to E4M3 in the kernel.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="row shard, one decode step",
            spec=dict(tokens=6, heads=16, local_heads=16, head_start=0, masked=True, rows=4096),
        ),
        Cell(
            why="head shard of 2: the upper half",
            spec=dict(tokens=96, heads=16, local_heads=8, head_start=8, masked=False, rows=4096),
        ),
        Cell(
            why="row shard, 2560 gathered attention-DP tokens",
            spec=dict(tokens=2560, heads=16, local_heads=16, head_start=0, masked=True, rows=65536),
        ),
        Cell(
            why="head shard, prefill tile of 8192 tokens",
            spec=dict(tokens=8192, heads=16, local_heads=8, head_start=0, masked=False, rows=65536),
        ),
        Cell(
            why="MXFP4 table (1017 export), row shard, one decode step",
            spec=dict(
                tokens=6, heads=16, local_heads=16, head_start=0, masked=True, rows=4096, mxfp4=True
            ),
        ),
        Cell(
            why="MXFP4 table, head shard of 2, 2560 gathered tokens",
            spec=dict(
                tokens=2560,
                heads=16,
                local_heads=8,
                head_start=8,
                masked=False,
                rows=65536,
                mxfp4=True,
            ),
        ),
    )
    note = (
        "Dequantization is exact (an E4M3 value, or an E2M1 value widened to E4M3, times a power of two) and "
        "rounds once to the output dtype."
    )

    def __call__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        ids: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
        *,
        head_start: int,
        block_size: int,
        masked: bool,
        vocab_start: int,
        vocab_end: int,
        num_ctas: int,
        mxfp4: bool = False,
    ) -> torch.Tensor:
        tokens, local_heads, dim = out.shape
        compile_engram_lookup(
            head_start, local_heads, ids.shape[1], dim, block_size, masked, out.dtype, mxfp4=mxfp4
        )(
            weight,
            scale,
            ids,
            offsets,
            out.view(tokens * local_heads, dim),
            vocab_start,
            vocab_end,
            num_ctas,
        )
        return out

    def reference(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        ids: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
        *,
        head_start: int,
        block_size: int,
        masked: bool,
        vocab_start: int,
        vocab_end: int,
        num_ctas: int,
        mxfp4: bool = False,
    ) -> torch.Tensor:
        tokens, local_heads, dim = out.shape
        row = ids[:, head_start : head_start + local_heads].long() + offsets.long()
        if masked:
            owned = (row >= vocab_start) & (row < vocab_end)
            row = torch.where(owned, row - vocab_start, torch.zeros_like(row))
        else:
            owned = torch.ones_like(row, dtype=torch.bool)
        rows = e2m1_to_e4m3_bytes(weight[row]) if mxfp4 else weight[row]
        values = rows.view(torch.float8_e4m3fn).double()
        exponent = (scale[row].long() - 127).double().exp2()
        values = values.unflatten(-1, (-1, block_size)) * exponent.unsqueeze(-1)
        values = values.flatten(-2).masked_fill(~owned.unsqueeze(-1), 0)
        return values.to(out.dtype)

    def is_valid(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        ids: torch.Tensor,
        offsets: torch.Tensor,
        out: torch.Tensor,
        *,
        head_start: int,
        block_size: int,
        masked: bool,
        vocab_start: int,
        vocab_end: int,
        num_ctas: int,
        mxfp4: bool = False,
    ) -> None:
        tokens, local_heads, dim = out.shape
        assert weight.dtype == scale.dtype == torch.uint8, "table bytes are passed as uint8"
        assert weight.shape[1] == (dim // 2 if mxfp4 else dim), (
            "a row is `dim` E4M3 bytes, or `dim / 2` E2M1 pairs"
        )
        assert scale.shape[1] * block_size == dim and dim % block_size == 0
        assert ids.dtype == torch.int64 and offsets.dtype == torch.int64
        assert ids.shape[0] == tokens and offsets.shape == (local_heads,)
        assert 0 <= head_start and head_start + local_heads <= ids.shape[1]
        assert out.is_contiguous() and out.is_cuda and num_ctas >= 1
        assert dim * out.element_size() % 16 == 0 and out.data_ptr() % 16 == 0
        assert weight.data_ptr() % 16 == 0, "the kernel assumes a 16-byte aligned table"
        assert not masked or vocab_start <= vocab_end


engram_lookup = EngramLookup()
