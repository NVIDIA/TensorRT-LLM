# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a DSA request-local KV position to its layer-independent latent-pool row, in CuTe DSL."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_req_index_to_global as _kernel
from .cute_dsl_req_index_to_global import (
    CompiledReqIndex,
    check_compiled,
    check_int32_range,
    pool_rows,
)

_GLM = dict(topk=2048, pool_layers=79)


def build(device_id: int, *, topk: int, max_blocks: int) -> CompiledReqIndex:
    """Compile the base-row launcher for `topk` columns and a `max_blocks`-wide block table.

    Same compile key and cost as `cute_dsl_req_index_to_global.build`; a
    separate launcher (the layer term is dropped at compile time).
    """
    with torch.cuda.device(device_id):
        fn = _kernel.compile_req_index_base_rawptr(device_id, int(topk), int(max_blocks))
    return CompiledReqIndex(fn=fn, topk=int(topk), max_blocks=int(max_blocks))


class CuteDslReqIndexToBase(OpWrapper):
    """Turn the indexer's request-local top-k positions into layer-independent pool rows.

    Returns a fresh int32 tensor shaped like `token_indices` holding

        block_table[req_id[i], p // 32] * stride_factor + p % 32

    for `p = token_indices[i, j]`, or -1 when `p < 0`, `p // 32 >=
    block_table.size(1)`, or the block-table entry is negative. Adding
    `layer * 32` to a valid entry gives `cute_dsl_req_index_to_global`'s row for
    that layer, which is what the sparse-MLA decode kernel does itself when told
    the index is a base row. No argument is modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target's decode step at MTP width 2 computes these once per fresh
    # selection instead of once per layer: up to 32 sequences x 2 rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one sequence at MTP 2, a context longer than top-k",
            spec=dict(**_GLM, seq_rows=[2], kv_lens=[4096], max_blocks=512),
        ),
        Cell(
            why="the full decode batch: 32 sequences x MTP 2",
            spec=dict(
                **_GLM,
                seq_rows=[2] * 32,
                kv_lens=[300 + 251 * i for i in range(32)],
                max_blocks=512,
            ),
        ),
        Cell(
            why="contexts shorter than top-k, positions past the table, unused pages",
            spec=dict(
                **_GLM, seq_rows=[2, 2], kv_lens=[33, 5000], max_blocks=64, pad_block_table=True
            ),
        ),
    )

    note = """
    `build(device_id, topk=, max_blocks=)` compiles the launcher; the target
    builds it on its first forward next to `cute_dsl_req_index_to_global`'s and
    passes it as `compiled`. Only the trunk's decode step at MTP width 2 calls
    it; other widths take the per-layer entry. `block_size` must be 32.
    """

    def __call__(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        stride_factor: int,
        *,
        compiled: CompiledReqIndex,
    ) -> torch.Tensor:
        return _kernel.convert_req_index_to_base_rawptr(
            req_id,
            block_table,
            token_indices,
            int(block_size),
            int(stride_factor),
            compiled.fn,
        )

    def reference(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        stride_factor: int,
        *,
        compiled: CompiledReqIndex,
    ) -> torch.Tensor:
        """The integer map in int64, so it cannot wrap; `compiled` is not read."""
        return pool_rows(req_id, block_table, token_indices, block_size, stride_factor, 0)

    def is_valid(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        stride_factor: int,
        *,
        compiled: CompiledReqIndex,
    ) -> None:
        check_compiled(compiled, block_table, token_indices)
        check_int32_range(block_table, stride_factor, 0)


cute_dsl_req_index_to_base = CuteDslReqIndexToBase()
