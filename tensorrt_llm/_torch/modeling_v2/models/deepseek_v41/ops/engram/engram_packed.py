# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram KV on packed MXFP8 rows, as stock's context path runs it (`DeepseekV41Engram.precompute_lookup` /
`project_lookup`): the table gather writes each token's owned heads as packed rows (their E4M3 bytes, then their UE8M0
codes, padded to 16 bytes) instead of dequantized bf16, and the WKV projection reads those rows straight from every
head shard (`EngramPackedProjectionKernel`) with the checkpoint's MXFP8 weight, so no bf16 embedding is built, reduced
or re-quantized.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.kernels.blackwell.engram.lookup import (
    compile_engram_lookup,
    packed_row_bytes,
)
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .engram_lookup import e2m1_to_e4m3_bytes


def _unpack(rows: torch.Tensor, heads: int, dim: int, block: int) -> torch.Tensor:
    """float64 `[M, heads * dim]` of packed rows `[M, row_bytes]`."""
    k = heads * dim
    vals = rows[:, :k].contiguous().view(torch.float8_e4m3fn).double()
    codes = rows[:, k : k + k // block].long()
    return (vals.unflatten(-1, (-1, block)) * (codes - 127).double().exp2().unsqueeze(-1)).flatten(
        -2
    )


class EngramLookupPacked(OpWrapper):
    """`ids` int64 `[T, num_heads]` bucket-local indices -> uint8 `[T, packed_row_bytes(local_heads, dim, block)]`
    packed rows of heads
    `[head_start, head_start + local_heads)` (head-sharded table: no ownership mask). With `mxfp4` the
    table rows are E2M1 pairs (`dim / 2` bytes), widened exactly to the same E4M3 packed rows."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="head shard of 2 (12 of 24 heads, dim 256): a context step's 9500 gathered tokens",
            spec=dict(
                tokens=9500, heads=24, local_heads=12, head_start=12, dim=256, rows=24 * 4096
            ),
        ),
        Cell(
            why="one decode token, first shard",
            spec=dict(tokens=1, heads=24, local_heads=12, head_start=0, dim=256, rows=24 * 4096),
        ),
        Cell(
            why="MXFP4 table (1017 export): head shard of 2, a context step's 9500 gathered tokens",
            spec=dict(
                tokens=9500,
                heads=24,
                local_heads=12,
                head_start=12,
                dim=256,
                rows=24 * 4096,
                mxfp4=True,
            ),
        ),
    )
    note = "A byte copy of the table's values and codes: the test compares bitwise."

    def __call__(
        self,
        weight,
        scale,
        ids,
        offsets,
        *,
        head_start: int,
        local_heads: int,
        block_size: int,
        num_ctas: int,
        mxfp4: bool = False,
    ) -> torch.Tensor:
        dim = weight.shape[1] * (2 if mxfp4 else 1)
        out = torch.empty(
            ids.shape[0],
            packed_row_bytes(local_heads, dim, block_size),
            dtype=torch.uint8,
            device=ids.device,
        )
        if ids.shape[0]:
            compile_engram_lookup(
                head_start,
                local_heads,
                ids.shape[1],
                dim,
                block_size,
                False,
                torch.uint8,
                packed=True,
                mxfp4=mxfp4,
            )(weight, scale, ids, offsets, out, 0, weight.shape[0], num_ctas)
        return out

    def reference(
        self,
        weight,
        scale,
        ids,
        offsets,
        *,
        head_start: int,
        local_heads: int,
        block_size: int,
        num_ctas: int,
        mxfp4: bool = False,
    ) -> torch.Tensor:
        dim = weight.shape[1] * (2 if mxfp4 else 1)
        row = (
            ids[:, head_start : head_start + local_heads].long() + offsets.long()
        )  # [T, local_heads]
        vals = (e2m1_to_e4m3_bytes(weight[row]) if mxfp4 else weight[row]).flatten(
            1
        )  # [T, local_heads * dim]
        codes = scale[row].flatten(1)  # [T, local_heads * dim / block]
        out = torch.zeros(
            ids.shape[0],
            packed_row_bytes(local_heads, dim, block_size),
            dtype=torch.uint8,
            device=ids.device,
        )
        out[:, : vals.shape[1]] = vals
        out[:, vals.shape[1] : vals.shape[1] + codes.shape[1]] = codes
        return out

    def is_valid(
        self,
        weight,
        scale,
        ids,
        offsets,
        *,
        head_start: int,
        local_heads: int,
        block_size: int,
        num_ctas: int,
        mxfp4: bool = False,
    ) -> None:
        dim = weight.shape[1] * (2 if mxfp4 else 1)
        assert weight.dtype == scale.dtype == torch.uint8 and scale.shape[1] * block_size == dim
        assert ids.dtype == offsets.dtype == torch.int64 and offsets.shape == (local_heads,)
        assert 0 <= head_start and head_start + local_heads <= ids.shape[1]


class EngramPackedProjection(OpWrapper):
    """`packed` uint8 `[S, M, row_bytes]` (source `s` holds features `[s * K/S, (s + 1) * K/S)` as packed rows) ->
    bf16 `[M, N]` = rows @ W.T, `weight` E4M3 `[N, K]` with its 1x32 UE8M0 scales `scale` `[N, K/32]` (uint8),
    `scale_sw` their CUTLASS-swizzled form and `alpha` fp32 `[1]` ones. `packed` may be a token slice of a gathered
    buffer (its strides are read)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep2 context step: 9500 tokens from 2 head shards, K 6144 -> N 25600",
            spec=dict(m=9500, sources=2, heads=24, dim=256, n=25600),
        ),
        Cell(
            why="a token slice of a 2-rank gathered buffer (strided)",
            spec=dict(m=700, sources=2, heads=24, dim=256, n=25600, slice=True),
        ),
        Cell(why="a short step", spec=dict(m=5, sources=2, heads=24, dim=256, n=25600)),
    )
    note = (
        "Exact operands (E4M3 x power-of-two scales) on both sides; "
        "fp32 accumulation order differs: gate 1e-2 rel RMS."
    )

    def __call__(self, packed, weight, scale, scale_sw, alpha) -> torch.Tensor:
        from tensorrt_llm._torch.kernels.blackwell.engram.packed_projection import (
            engram_packed_projection,
        )

        out = torch.empty(
            packed.shape[1], weight.shape[0], dtype=torch.bfloat16, device=packed.device
        )
        return engram_packed_projection(packed, weight, scale_sw, alpha, out=out)

    def reference(self, packed, weight, scale, scale_sw, alpha, block: int = 32) -> torch.Tensor:
        s, m, _ = packed.shape
        n, k = weight.shape
        local_k = k // s
        x = torch.cat([_unpack(packed[i], 1, local_k, block) for i in range(s)], 1)  # [M, K]
        w = weight.double() * (scale.long() - 127).double().exp2().repeat_interleave(block, 1)
        return (x @ w.t() * alpha.double()).to(torch.bfloat16)

    def is_valid(self, packed, weight, scale, scale_sw, alpha) -> None:
        s, m, rb = packed.shape
        n, k = weight.shape
        assert packed.dtype == torch.uint8 and packed.stride(2) == 1 and packed.stride(1) % 16 == 0
        assert k % s == 0 and rb >= k // s + k // s // 32 and weight.dtype == torch.float8_e4m3fn
        assert scale.shape == (n, k // 32) and alpha.dtype == torch.float32 and alpha.numel() == 1


engram_lookup_packed = EngramLookupPacked()
engram_packed_projection = EngramPackedProjection()
