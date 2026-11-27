# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generation rows' sparse MLA as the target calls it: `decode_slots` (a pure decode
step's selection as the attention reads it, per layer) and `sparse_mla_decode` (the
attention), each picking its kernel by the handle built for the engine at its first forward.

A pure decode step at MTP3 / MTP5's width runs the union attention where it planned the
batch: it reads the indexer's request-local positions directly, so no conversion runs. At a
width whose decode grid takes them the selection converts once per fresh selection into
layer-independent base rows (the kernel adds the layer's offset), else into each layer's
global pool rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import OpWrapper

from .._build import Count
from .cute_dsl_req_index_to_base import cute_dsl_req_index_to_base
from .cute_dsl_req_index_to_global import cute_dsl_req_index_to_global
from .cute_dsl_sparse_mla_decode import (
    INDEX_KIND_BASE_WITH_LAYER_OFFSET,
    INDEX_KIND_GLOBAL,
    cute_dsl_sparse_mla_decode,
)
from .cute_dsl_sparse_mla_decode_union import cute_dsl_sparse_mla_decode_union

#: The MTP width the width-2 decode grid serves (MTP1); the union attention serves the
#: widths above it.
MTP1_WIDTH = 2

#: The generation index kind of a pure decode step whose attention takes the union kernel:
#: the slots are the step's request-local selection (`UnionSlots`).
UNION_POSITIONS = "union_positions"


class UnionSlots(NamedTuple):
    """A pure decode step's generation selection as the union attention takes it: the
    request-local top-k positions `[rows, 2048]` and each row's block-table row."""

    positions: torch.Tensor
    req_idx: torch.Tensor


@dataclass(frozen=True)
class SparseMlaHandle:
    """The engine's decode attention: `grid` the width-2 handle (MTP1, no speculation, the
    1-row draft steps), `wide` the MTP3 / MTP5 width's grid (None otherwise), `union` the
    union attention (None: per-row only); `req_global` / `req_base` the selection
    conversions; `stride_factor` the pool's row stride per layer; `widths` the decode widths
    whose grids take base rows (`mtp_widths`)."""

    grid: Any
    wide: Any
    union: Any
    req_global: Any
    req_base: Any
    stride_factor: int
    widths: frozenset
    count: Count


def decode_grid(gen_p: int, handle: SparseMlaHandle):
    """The decode attention's grid for `gen_p` rows per sequence: the width-2 grid up to its
    width, the MTP3 / MTP5 grid above."""
    if gen_p <= handle.grid.max_sq:
        return handle.grid
    return handle.wide


class DecodeSlots(OpWrapper):
    """A pure decode step's selection `buf` as pool layer `layer_id`'s sparse MLA decode reads
    it: `(selection, index kind, slot layer)`. `memo` is the step's dict (the Target's, one per
    step): the step's kind is decided at its first call (the union attention where it planned
    `md.num_seqs` at this width, else base rows where the width's grid takes them, else
    global rows), and base rows, being layer-independent, are recomputed only where a layer
    selected afresh (`fresh`)."""

    def __call__(
        self,
        md,
        req_idx: torch.Tensor,
        buf: torch.Tensor,
        fresh: bool,
        layer_id: int,
        gen_p: int,
        memo: dict,
        handle: SparseMlaHandle,
    ):
        kind = memo.get("kind")
        if kind is None:
            union = handle.union
            if (
                union is not None
                and MTP1_WIDTH < gen_p <= union.max_sq
                and md.num_seqs in union.plans
            ):
                kind = UNION_POSITIONS
            elif (
                gen_p in handle.widths
                and INDEX_KIND_BASE_WITH_LAYER_OFFSET in decode_grid(gen_p, handle).index_kinds
            ):
                kind = INDEX_KIND_BASE_WITH_LAYER_OFFSET
            else:
                kind = INDEX_KIND_GLOBAL
            memo["kind"] = kind
        if kind == UNION_POSITIONS:
            return UnionSlots(buf, req_idx), UNION_POSITIONS, layer_id
        if kind == INDEX_KIND_GLOBAL:
            rows = cute_dsl_req_index_to_global(
                req_idx, md.block_table, buf, md.tokens_per_block, handle.stride_factor,
                layer_id, compiled=handle.req_global,
            )  # fmt: skip
            return rows, INDEX_KIND_GLOBAL, None
        if fresh or "base" not in memo:
            memo["base"] = cute_dsl_req_index_to_base(
                req_idx, md.block_table, buf, md.tokens_per_block, handle.stride_factor,
                compiled=handle.req_base,
            )  # fmt: skip
        return memo["base"], INDEX_KIND_BASE_WITH_LAYER_OFFSET, layer_id

    def reference(
        self,
        md,
        req_idx: torch.Tensor,
        buf: torch.Tensor,
        fresh: bool,
        layer_id: int,
        gen_p: int,
        memo: dict,
        handle: SparseMlaHandle,
    ) -> torch.Tensor:
        """Each selected position's global pool row of layer `layer_id` in int64 (the union
        and base kinds are the same rows before the kernel adds the layer offset)."""
        return cute_dsl_req_index_to_global.reference(
            req_idx, md.block_table, buf, md.tokens_per_block, handle.stride_factor, layer_id,
            compiled=handle.req_global,
        )  # fmt: skip


class SparseMlaDecode(OpWrapper):
    """The sparse MLA decode of the generation rows' e4m3 query into `lat_out`: on the union
    kernel where the step's selection is the union attention's (`slots` from `decode_slots`,
    counted `fmha_dec_union`), else on the decode grid for `gen_p` rows per sequence (the
    MTP3 / MTP5 grid counted `fmha_dec_wide`). `slots` is `(selection, index kind, slot
    layer)`; `ctx` the step's context request count."""

    def __call__(
        self,
        quant_q: torch.Tensor,
        md,
        slots,
        lat_out: torch.Tensor,
        cu_q: torch.Tensor,
        bmm1: torch.Tensor,
        bmm2: torch.Tensor,
        ctx: int,
        gen_p: int,
        handle: SparseMlaHandle,
    ) -> None:
        sel, kind, slot_layer = slots
        seqs = int(quant_q.shape[0]) // gen_p
        pool = md.kv_cache_manager.get_unique_primary_pool()
        kv_lens = md.kv_lens_cuda_runtime[ctx : ctx + seqs]
        if kind == UNION_POSITIONS:
            handle.count("fmha_dec_union", True)
            cute_dsl_sparse_mla_decode_union(
                quant_q, pool, sel.positions, md.block_table, sel.req_idx, lat_out,
                cu_q[: seqs + 1], kv_lens, slot_layer, handle.stride_factor,
                compiled=handle.union,
            )  # fmt: skip
            return
        grid = decode_grid(gen_p, handle)
        if grid is handle.wide:
            handle.count("fmha_dec_wide", True)
        cute_dsl_sparse_mla_decode(
            quant_q, pool, sel, lat_out, cu_q[: seqs + 1], kv_lens, bmm1, bmm2, kind,
            slot_layer, compiled=grid,
        )  # fmt: skip

    def reference(
        self,
        quant_q: torch.Tensor,
        md,
        slots,
        lat_out: torch.Tensor,
        cu_q: torch.Tensor,
        bmm1: torch.Tensor,
        bmm2: torch.Tensor,
        ctx: int,
        gen_p: int,
        handle: SparseMlaHandle,
    ) -> torch.Tensor:
        """The per-row kernel's reference (`cute_dsl_sparse_mla_decode.reference`) over the
        same selection: the union kernel computes the same attention per row."""
        sel, kind, slot_layer = slots
        assert kind != UNION_POSITIONS, "the union selection's reference is the union entry's"
        seqs = int(quant_q.shape[0]) // gen_p
        return cute_dsl_sparse_mla_decode.reference(
            quant_q, md.kv_cache_manager.get_unique_primary_pool(), sel, lat_out,
            cu_q[: seqs + 1], md.kv_lens_cuda_runtime[ctx : ctx + seqs], bmm1, bmm2, kind,
            slot_layer, decode_grid(gen_p, handle),
        )  # fmt: skip


decode_slots = DecodeSlots()
sparse_mla_decode = SparseMlaDecode()
