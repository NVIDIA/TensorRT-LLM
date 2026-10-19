# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA index side-cache store: raw index keys and position coordinates into their pages."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's index geometry and the engine's QSA metadata. `token_k` is
#: the k head of the packed `[T, 4 + 1, 128]` index projection; the coordinates
#: are the `[T, 3]` transpose of the engine's `[3, max_num_tokens]` position
#: buffer; `slots` x `width` is `qsa_block_table`. `pages` is the pool.
_CKPT = dict(
    hkv=1,
    head_dim=128,
    index_heads=4,
    tokens_per_block=32,
    axes=3,
    slots=32,
    width=260,
    key_dtype=torch.bfloat16,
    request_dtype=torch.int32,
    position_dtype=torch.int64,
    key_layout="packed",
    coords_layout="transposed",
    meta_layout="contiguous",
    cache_layout="plain",
    table_layout="dense",
)

_DECODE = ((0, 1), (1, 1), (31, 1), (32, 1), (33, 1), (63, 1), (64, 1), (95, 1))


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.view(as_int)


class QsaPagedIndexStore(OpWrapper):
    """Write each row's raw index key and position coordinates into its paged slot.

    For row t, with `page = block_table[request_indices[t], logical_positions[t]
    // tokens_per_block]` and `slot = logical_positions[t] % tokens_per_block`::

        index_cache[page, slot] = token_k[t]  # cast to the cache dtype
        position_cache[page, slot, :A] = position_coordinates[t]  # A = coordinates' width

    The key is stored raw -- un-normed, un-rotated: `qsa_prefill_compress`
    pools raw keys and applies the Gemma norm and the rotation itself. Returns
    `(index_cache, position_cache)`, both written in place.

    Sourcing tier 3, composed from native torch: a block-table gather, a
    floor-divide and remainder, two indexed writes (8 launches). Nothing in
    the engine performs this store for a packed, ragged batch: the registered
    index-cache ops write fp8 planes or norm and rotate on the way in,
    `triton_qsa_decode_pre_indexer` is one row per request, and
    `QSAIndexer.update_cache_and_compress` bundles the compression.

    Fusion boundary: page lookup, slot arithmetic, both scatters. The index
    projection, the query's norm and rotation, group compression, scoring,
    selection, sparse attention and the main K/V pool are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: 8 rows at aligned, mid and page-crossing positions",
            spec=_spec(_DECODE, pages=32),
        ),
        Cell(
            why="decode MTP: 4 requests x 2 rows, a pair straddling a page",
            spec=_spec(((10, 2), (31, 2), (63, 2), (5, 2)), pages=16),
        ),
        Cell(
            why="prefill: 1024 rows, 4 requests x 256 walking 8 pages each",
            spec=_spec(((0, 256),) * 4, pages=32),
        ),
        Cell(
            why="mixed: two context chunks and two decode rows, 66 rows",
            spec=_spec(((0, 40), (0, 24), (200, 1), (7, 1)), pages=32),
        ),
        Cell(
            why="contiguous token_k and coordinates",
            spec=_spec(_DECODE, pages=32, key_layout="contiguous", coords_layout="contiguous"),
        ),
        Cell(
            why="text batch: stride-0 expand of one position column",
            spec=_spec(((0, 1), (5, 1), (32, 1), (70, 1)), pages=16, coords_layout="stride0"),
        ),
        Cell(
            why="a 5-axis position plane: axes past the 3rd untouched",
            spec=_spec(((2, 1), (33, 1)), pages=8, axes=5),
        ),
        Cell(
            why="metadata as columns of wider buffers (strided)",
            spec=_spec(((0, 8),), pages=8, meta_layout="strided"),
        ),
        Cell(
            why="index_cache as a role view of a wider allocation",
            spec=_spec(((0, 1), (31, 1), (32, 1), (33, 1)), pages=8, cache_layout="role_view"),
        ),
        Cell(
            why="block table as a column slice of a wider one",
            spec=_spec(((0, 1), (1, 1), (32, 1), (33, 1)), pages=8, table_layout="column_slice"),
        ),
        Cell(
            why="fp16 keys: one conversion into the bf16 plane",
            spec=_spec(((0, 4),), pages=8, key_dtype=torch.float16),
        ),
        Cell(
            why="fp32 keys: one conversion into the bf16 plane",
            spec=_spec(((0, 4),), pages=8, key_dtype=torch.float32),
        ),
        Cell(
            why="int32 requests and int32 positions",
            spec=_spec(_DECODE, pages=32, position_dtype=torch.int32),
        ),
        Cell(
            why="int64 requests and int64 positions",
            spec=_spec(_DECODE, pages=32, request_dtype=torch.int64),
        ),
        Cell(why="zero rows: both planes unchanged", spec=_spec((), pages=8)),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_req_idx_per_token` (int32) and `qsa_logical_positions` (int64)
    sliced to the step's rows, `qsa_block_table` unsliced, the layer's
    `get_index_k_buffer(i)` (`[pages, tpb, 1, 128]` bf16) and the shared
    `get_qsa_position_buffer()` (`[pages, tpb, 3]` int32, rewritten
    idempotently by every sparse layer). Every operand is read through its
    own strides, unlike the Triton siblings.

    Values the caller owns -- data, not metadata, so not guarded; each
    measured, and unlike `qsa_paged_kv_store` nothing is skipped:
      - a `-1` (unallocated) table entry wraps to the pool's last page, and a
        negative request selects the table's last row;
      - a negative position wraps twice (last table column, last slot);
      - a page, request, column or slot genuinely out of range is a CUDA
        device-side assert that poisons the process -- so none is tested;
      - duplicate destinations both store, unordered;
      - coordinates wider than the plane's dtype wrap (int64 `2^31 + 5`
        stores as -2147483643); floats truncate toward zero.
    Loud rejections are torch's own: a 1-D table, a position plane narrower
    than the coordinates, mixed devices, non-index metadata dtypes.
    `rows == 0` leaves both planes unchanged.

    The 8 launches are launch-bound: ~0.15 ms per call flat from 1 to 1024
    rows. CUDA-graph capturable; a replay follows positions rewritten in
    place. Repeating a call is idempotent.

    Hard limits: torch advanced indexing in int64 -- no int32 boundary inside
    the call; the coordinate wrap above is the plane's dtype.

    Not certified: an fp8 index plane, `Hkv` above 1, multiple devices,
    concurrent streams onto one pool.
    """

    def __call__(
        self,
        token_k: torch.Tensor,
        position_coordinates: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = logical_positions.to(torch.long)
        pages = block_table[request_indices.to(torch.long), positions // tokens_per_block].to(
            torch.long
        )
        within = positions % tokens_per_block
        index_cache[pages, within] = token_k.to(index_cache.dtype)
        position_cache[pages, within, : position_coordinates.shape[-1]] = position_coordinates.to(
            position_cache.dtype
        )
        return index_cache, position_cache

    def reference(
        self,
        token_k: torch.Tensor,
        position_coordinates: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both planes after the store, as fresh tensors, built row by row.

        Pages and slots are computed in Python and each write is a basic
        (integer) index -- none of the advanced indexing `__call__` is made of.
        """
        expected_index, expected_position = index_cache.clone(), position_cache.clone()
        table = block_table.tolist()
        axes = position_coordinates.shape[-1]
        rows = zip(request_indices.tolist(), logical_positions.tolist())
        for row, (request, position) in enumerate(rows):
            page = table[request][position // tokens_per_block]
            slot = position % tokens_per_block
            expected_index[page, slot] = token_k[row].to(index_cache.dtype)
            expected_position[page, slot, :axes] = position_coordinates[row].to(
                position_cache.dtype
            )
        return expected_index, expected_position

    def is_valid(
        self,
        token_k: torch.Tensor,
        position_coordinates: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> None:
        # Each is a shape check over a violation measured to be accepted and
        # silently wrong: torch broadcasts instead of raising.
        rows = request_indices.shape[0]
        assert request_indices.dim() == 1 and logical_positions.dim() == 1, (
            f"request_indices {tuple(request_indices.shape)} and logical_positions "
            f"{tuple(logical_positions.shape)} must both be one-dimensional; a "
            f"trailing axis broadcasts into a rows x rows index instead of raising"
        )
        assert logical_positions.shape[0] == rows, (
            f"logical_positions has {logical_positions.shape[0]} entries against "
            f"{rows} request_indices; a length-1 operand broadcasts silently and a "
            f"length-1 logical_positions collapses every row onto one slot"
        )
        assert token_k.shape[0] == rows, (
            f"token_k has {token_k.shape[0]} rows against {rows} metadata rows; a "
            f"single row broadcasts into every destination slot instead of raising"
        )
        assert position_coordinates.shape[0] == rows, (
            f"position_coordinates has {position_coordinates.shape[0]} rows against "
            f"{rows} metadata rows; a single row broadcasts into every slot"
        )
        assert index_cache.shape[1] == tokens_per_block, (
            f"tokens_per_block={tokens_per_block} does not match index_cache's "
            f"token axis {index_cache.shape[1]}; a wrong page size writes the "
            f"wrong page and the wrong slot"
        )
        assert position_cache.shape[1] == tokens_per_block, (
            f"tokens_per_block={tokens_per_block} does not match position_cache's "
            f"token axis {position_cache.shape[1]}"
        )
        assert index_cache.shape[2] == token_k.shape[1], (
            f"index_cache carries {index_cache.shape[2]} index-KV heads against "
            f"token_k's {token_k.shape[1]}; a one-head token_k broadcasts into "
            f"every head of a wider cache"
        )


qsa_paged_index_store = QsaPagedIndexStore()
