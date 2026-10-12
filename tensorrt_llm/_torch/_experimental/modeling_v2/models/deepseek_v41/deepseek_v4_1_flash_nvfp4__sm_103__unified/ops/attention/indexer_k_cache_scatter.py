# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paged index-K cache write, via the trtllm op.

The op is `torch.ops.trtllm.indexer_k_cache_scatter_op`, registered from C++
(`cpp/tensorrt_llm/thop/IndexerKCacheScatterOp.cpp`, kernel in
`cpp/tensorrt_llm/kernels/indexerKCacheScatter.cu`).
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Page sizes in compressed entries at tokens_per_block 128: 64 is the ratio-2
#: page, 128 the ratio-1 page.
_BLOCK_KV = (64, 128)


def _steps(block_kv: int) -> tuple[tuple[str, dict], ...]:
    """(why, spec) of the DeepSeek-V4.1 steps the target writes, at one page size.

    `cached` / `new` are compressed entries per request; `cached_below` /
    `new_upto` with `requests` draw them at random (seeded by the test).
    """
    r = 128 // block_kv
    return (
        ("decode B=1, one entry mid-page", dict(cached=[3 * block_kv + 5], new=[1])),
        (
            "decode B=32, one entry each, cached < 8 pages",
            dict(requests=32, cached_below=8 * block_kv, new=[1] * 32),
        ),
        (
            f"DSpark verify B=32, 0..{6 // r} entries each",
            dict(requests=32, cached_below=8 * block_kv, new_upto=6 // r),
        ),
        (f"prefill of 8192 tokens, {8192 // r} entries", dict(cached=[0], new=[8192 // r])),
        (f"prefill of 16384 tokens, {16384 // r} entries", dict(cached=[0], new=[16384 // r])),
        (
            "chunked mixed: mid-page continuation, page edge",
            dict(cached=[1003, 0, 64 * 3 - 1], new=[3000, 77, 2]),
        ),
    )


class IndexerKCacheScatter(OpWrapper):
    """Copy rows `0 .. num_tokens-1` of packed index keys and their scale words into a paged pool.

    One call, one kernel launch, no arithmetic. For each token `t`:

        k_cache.flat[slot_mapping_data[t]  : + D] = k_data[t, :]    # D = k_data.size(1) bytes
        k_cache.flat[slot_mapping_scale[t] : + 4] = bytes of k_scale[t, 0]

    where `.flat` is `k_cache`'s logical `[pages, block_kv, 1, 68]` shape
    unravelled innermost-first and addressed through its strides (a page-strided
    view is written correctly). A token whose data or scale offset is negative is
    skipped entirely; every other byte of the pool is left as it was.

    MXFP4 mode: `k_data` uint8 `[T, 64]` (two E2M1 codes per byte, element `2i`
    in the low nibble of byte `i`), `k_scale` int32 `[T, 1]` (four UE8M0 bytes,
    byte `g` scaling elements `32g .. 32g+31`), 68-byte pool entries.

    Fusion boundary: the copy only. The caller owns the index-K projection,
    norm and RoPE, the quantization, the page table, the offsets and the pool's
    allocation. Writes `k_cache` in place and returns it.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(why=f"block_kv {bk}: {why}", spec=dict(block_kv=bk, **spec))
            for bk in _BLOCK_KV
            for why, spec in _steps(bk)
        )
        + tuple(
            Cell(
                why=f"skip {which}: -1 offsets on every 5th row skip the whole token",
                spec=dict(block_kv=64, cached=[5, 130, 0], new=[40, 17, 64], skip=which),
            )
            for which in ("both", "data_only", "scale_only")
        )
        + (
            Cell(
                why="num_tokens 150 of 256 rows: the buffer tail is never read",
                spec=dict(block_kv=128, cached=[7, 300], new=[200, 56], num_tokens=150),
            ),
            Cell(
                why="num_tokens 0: a no-op on a populated pool",
                spec=dict(block_kv=128, cached=[7, 300], new=[200, 56], num_tokens=0),
            ),
            Cell(
                why="page-strided pool view, stride(0) = 64 * 68 + 128",
                spec=dict(block_kv=64, cached=[10, 200], new=[100, 33], page_pad=128),
            ),
        )
    )

    note = """
    Cells: keys are bf16 [T, 128] with a different power-of-two magnitude per
    32-block, quantized by `torch.ops.trtllm.fp4_quantize(k, None, 32, True,
    False)`; its data passes as is and its linear 1-D uint8 scales as
    `sf.view(torch.int32).view(T, 1)`, a reinterpretation. Pools hold 3x the
    pages a batch needs plus 8, pages drawn by a random permutation.

    Metadata consumed: none. The caller builds the offsets; for the page-blocked
    layout the DeepGEMM paged indexer-logits kernel reads (`block_kv x 64` code
    bytes, then `block_kv x 4` scale bytes per page) they are
        slot_mapping_data  = page * block_kv * 68 + slot * 64
        slot_mapping_scale = page * block_kv * 68 + block_kv * 64 + slot * 4
    The op cannot tell a right layout from a wrong one: a rolled page table,
    key-interleaved offsets, 64-entry pages addressed as 128, byte-reversed or
    group-major scale bytes, and another owner's view are all accepted and
    measured wrong on every row (byte-reversed scales: 99%).

    Preconditions: pass the KV-source owner's own pool view -- the ratio-2
    owners (layers 2, 8, 14) interleave their pages in one allocation, so a page
    id means something only through its own owner's view. Offsets must be in
    range and multiples of 4: the kernel neither bounds-checks nor aligns them
    (read, not probed). Two tokens given overlapping offsets race. A `-1`
    page-table column yields negative offsets and is skipped, but mask such rows
    to `-1` explicitly. The op rejects loudly a non-uint8, non-4-D or
    `size(2) != 1` pool, a `k_scale` that is not 2-D with 4-byte elements,
    non-int64 slot mappings, a non-contiguous `k_data`, CPU tensors, and data
    widths other than 64 and 128.

    CUDA graphs: captures and replays correctly; `num_tokens` is baked into the
    graph, so pad with `-1` offsets rather than changing it. Every `num_tokens`
    in 1..16384 is certified eagerly at both page sizes; captured at n = G and,
    for the DSpark verify step, 6G (ratio 1) and 3G (ratio 2), for G in 1..32,
    40..128 step 8, 192, 256. One precompiled kernel serves every row count.

    Pools: any size up to 6,539,977 blocks (the largest any DeepSeek-V4.1 cache
    manager can have on a GB300), the far end past 2^35 bytes included; the
    kernel forms 64-bit byte offsets. A caller that wraps offsets to 32 bits
    gets silently wrong placement.

    Not certified: other architectures; the FP8 mode (128-byte rows, float32
    scale, 132-byte entries); `block_kv` other than 64 and 128; more than 16384
    rows in one call; pools above 6,539,977 blocks; out-of-range, misaligned or
    overlapping offsets; capture at row counts outside the sets above; graph
    replay on a live cache manager's own metadata buffers.
    """

    def raw_call(
        self,
        k_data: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_data: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.indexer_k_cache_scatter_op(
            k_data, k_scale, k_cache, slot_mapping_data, slot_mapping_scale, num_tokens
        )
        return k_cache

    def is_valid(
        self,
        k_data: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_data: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> None:
        # Both guards are pure metadata and cover inputs the op was observed to accept silently:
        # 128-byte FP8-width rows written into the 68-byte-per-entry FP4 pool overwrite the
        # neighbouring entries, and a num_tokens larger than the rows or slot entries supplied
        # reads past the end of those tensors.
        assert k_data.size(-1) + 4 == k_cache.size(-1), (
            "k_data row bytes + 4 scale bytes must equal the pool's per-entry bytes "
            "(64 + 4 = 68 for MXFP4, 128 + 4 = 132 for FP8)"
        )
        assert num_tokens <= min(
            k_data.size(0), k_scale.size(0), slot_mapping_data.numel(), slot_mapping_scale.numel()
        ), "num_tokens must not exceed the rows of k_data / k_scale or the slot-mapping lengths"

    def reference(
        self,
        k_data: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_data: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """The pool after the call, placed by native torch indexing on flat byte offsets.

        `k_cache` is the pool as it was before the call: the op writes it in
        place, so the test passes a snapshot taken before calling it. Nothing is
        mutated; the result is a new contiguous tensor of `k_cache`'s logical
        shape. Each scale element's bytes are placed in memory order, as the op
        copies them.
        """
        expected = k_cache.clone(memory_format=torch.contiguous_format)
        flat = expected.view(-1)
        n = num_tokens
        d_off = slot_mapping_data[:n].long()
        s_off = slot_mapping_scale[:n].long()
        keep = (d_off >= 0) & (s_off >= 0)
        data_bytes = k_data.size(1) * k_data.element_size()
        scale_bytes = k_scale.size(1) * k_scale.element_size()
        data = k_data[:n].contiguous().view(torch.uint8).reshape(n, data_bytes)[keep]
        scale = k_scale[:n].contiguous().view(torch.uint8).reshape(n, scale_bytes)[keep]
        dev = flat.device
        flat[d_off[keep].unsqueeze(1) + torch.arange(data.size(1), device=dev)] = data
        flat[s_off[keep].unsqueeze(1) + torch.arange(scale.size(1), device=dev)] = scale
        return expected

    def compare(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        """Bitwise over the whole pool.

        The op is a byte copy, so there is nothing to round: any byte that
        differs is a row written to the wrong place, a skipped row written, or a
        byte outside the written ranges touched.
        """
        assert out.dtype == ref.dtype == torch.uint8, (out.dtype, ref.dtype)
        assert out.shape == ref.shape, (out.shape, ref.shape)
        bad = (out != ref).sum().item()
        assert bad == 0, f"{bad} of {out.numel()} pool bytes differ"


indexer_k_cache_scatter = IndexerKCacheScatter()
