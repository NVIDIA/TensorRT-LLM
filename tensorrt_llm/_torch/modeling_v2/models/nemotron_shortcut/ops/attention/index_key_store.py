# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write the sparse-attention indexer's key of each token into its page of the index-key cache."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


@triton.jit
def _index_key_store_kernel(
    keys,
    keys_stride,
    cache,
    cache_slot_stride,
    cache_token_stride,
    offsets,
    offsets_stride,
    row_req,
    row_pos,
    slot_scale,
    TPB: tl.constexpr,
    D: tl.constexpr,
):
    r = tl.program_id(0)
    b = tl.load(row_req + r)
    pos = tl.load(row_pos + r)
    page_offset = tl.load(offsets + b.to(tl.int64) * offsets_stride + pos // TPB)
    slot = (page_offset // slot_scale).to(tl.int64)
    d = tl.arange(0, D)
    key = tl.load(keys + r.to(tl.int64) * keys_stride + d)
    tl.store(cache + slot * cache_slot_stride + (pos % TPB) * cache_token_stride + d, key)


class IndexKeyStore(OpWrapper):
    """In place: `cache[k_offsets[row_req[r], row_pos[r] // tpb] // slot_scale,
    row_pos[r] % tpb, 0, :] = keys[r]` for every row `r`. Returns None.

    `keys` `[N, D]` bf16 with unit last-dim stride (a column slice of a wider
    projection is fine). `cache` is the layer's index-key view
    `[slots, tpb, 1, D]` (`get_index_k_buffer`), whose slot
    dimension may be strided. `k_offsets` `[B, max_blocks]` int32 is the K
    plane of the attention pool's block offsets, `kv_cache_block_offsets[pool,
    :, 0]` -- one lifecycle slot times the pool's page-index scale per page --
    so the slot of a page is its offset divided by `slot_scale`. `row_req` and
    `row_pos` (int32 `[N]`) name each row's request and absolute position;
    the page holding the position must be allocated.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode rows, 32-token pages",
            spec=dict(tpb=32, rows=(1, 1, 1, 1), past=(5, 40, 97, 300)),
        ),
        Cell(
            why="short prefill chunks, 64-token pages",
            spec=dict(tpb=64, rows=(3, 3, 3), past=(61, 128, 999)),
        ),
        Cell(
            why="prefill chunks after a prefix, 128-token pages",
            spec=dict(tpb=128, rows=(700, 33), past=(0, 400)),
        ),
    )

    note = """
    The slot arithmetic is the KV cache manager V2's: its device block offsets
    are `slot * index_scale` for K (plus `kv_offset` for V), and the
    index-key view is indexed by the same lifecycle slot. Rows are written
    independently, so duplicate (request, position) pairs race.
    """

    def __call__(
        self,
        keys: torch.Tensor,
        cache: torch.Tensor,
        k_offsets: torch.Tensor,
        slot_scale: int,
        row_req: torch.Tensor,
        row_pos: torch.Tensor,
    ) -> None:
        n, d = keys.shape
        if n == 0:
            return
        _index_key_store_kernel[(n,)](
            keys,
            keys.stride(0),
            cache,
            cache.stride(0),
            cache.stride(1),
            k_offsets,
            k_offsets.stride(0),
            row_req,
            row_pos,
            slot_scale,
            TPB=cache.shape[1],
            D=d,
        )

    def reference(self, keys, cache, k_offsets, slot_scale, row_req, row_pos) -> torch.Tensor:
        """The cache after the writes; nothing is mutated."""
        out = cache.clone()
        tpb = cache.shape[1]
        for r, (b, pos) in enumerate(zip(row_req.tolist(), row_pos.tolist())):
            slot = int(k_offsets[b, pos // tpb]) // slot_scale
            out[slot, pos % tpb, 0] = keys[r]
        return out

    def is_valid(self, keys, cache, k_offsets, slot_scale, row_req, row_pos) -> None:
        assert (
            keys.dim() == 2
            and keys.stride(1) == 1
            and keys.dtype == cache.dtype
            and cache.dim() == 4
            and cache.shape[2] == 1
            and cache.shape[3] == keys.shape[1]
            and cache.stride(3) == 1
            and cache.stride(1) == cache.shape[3]
            and k_offsets.dtype == torch.int32
            and k_offsets.stride(1) == 1
            and row_req.dtype == torch.int32
            and row_pos.dtype == torch.int32
            and row_req.numel() == keys.shape[0] == row_pos.numel()
            and slot_scale > 0
        ), "IndexKeyStore received unsupported inputs"


index_key_store = IndexKeyStore()
