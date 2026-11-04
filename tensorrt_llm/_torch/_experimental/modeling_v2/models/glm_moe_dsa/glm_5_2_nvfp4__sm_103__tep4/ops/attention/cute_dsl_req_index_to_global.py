# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a DSA request-local KV position to its row in the paged latent pool, in CuTe DSL."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_req_index_to_global as _kernel

#: The page size the kernel is specialised for (it rejects any other).
BLOCK_SIZE = 32

#: GLM-5.2's selection width and the pool layer count of the trunk (78 + MTP).
_GLM = dict(topk=2048, pool_layers=79)


@dataclass(frozen=True)
class CompiledReqIndex:
    """A compiled launcher and the two shapes it bakes into its addressing."""

    fn: Callable
    topk: int
    max_blocks: int


def build(device_id: int, *, topk: int, max_blocks: int) -> CompiledReqIndex:
    """Compile the per-layer launcher for `topk` columns and a `max_blocks`-wide block table.

    Both are compile-time constants: `topk` is the selection width and must be a
    multiple of 256, `max_blocks` is the engine's block-table width. One compile
    of about a second, cached per process; allocates nothing. PDL launch.
    """
    with torch.cuda.device(device_id):
        fn = _kernel.compile_req_index_rawptr(device_id, int(topk), int(max_blocks))
    return CompiledReqIndex(fn=fn, topk=int(topk), max_blocks=int(max_blocks))


def check_compiled(compiled, block_table: torch.Tensor, token_indices: torch.Tensor) -> None:
    """The guard both index entries share: the call's shapes are the compiled ones."""
    # `block_table` is read at a row stride of the compiled `max_blocks` and
    # `token_indices` / the output at one of the compiled `topk`; a call of
    # another width reads and writes the wrong rows.
    assert block_table.shape[1] == compiled.max_blocks, (
        f"block_table is {block_table.shape[1]} blocks wide; compiled for {compiled.max_blocks}"
    )
    assert token_indices.shape[1] == compiled.topk, (
        f"token_indices has {token_indices.shape[1]} columns; compiled for {compiled.topk}"
    )


def check_int32_range(block_table: torch.Tensor, stride_factor: int, layer_id: int) -> None:
    """Pool rows are computed in int32 on the device: past 2**31 they wrap silently."""
    if block_table.numel():
        top = int(block_table.max()) * stride_factor + (layer_id + 1) * BLOCK_SIZE
        assert top < 2**31, f"pool row {top} does not fit int32; the kernel wraps it silently"


def pool_rows(
    req_id: torch.Tensor,
    block_table: torch.Tensor,
    token_indices: torch.Tensor,
    block_size: int,
    stride_factor: int,
    layer_offset: int,
) -> torch.Tensor:
    """`block_table[req, p // bs] * stride + p % bs + layer_offset`, or -1, in int64."""
    tok = token_indices.long()
    max_blocks = block_table.shape[1]
    block = torch.div(tok, block_size, rounding_mode="floor")
    valid = (tok >= 0) & (block < max_blocks)
    rows = req_id.long().unsqueeze(1).expand_as(tok)
    base = block_table.long()[rows, block.clamp(0, max_blocks - 1)]
    valid &= base >= 0
    mapped = base * stride_factor + (tok - block * block_size) + layer_offset
    return torch.where(valid, mapped, torch.full_like(tok, -1)).to(torch.int32)


class CuteDslReqIndexToGlobal(OpWrapper):
    """Turn the indexer's request-local top-k positions into this layer's latent-pool rows.

    Entry `token_indices[i, j]` is a KV position `p` from the start of request
    `req_id[i]`'s sequence. The op returns a fresh int32 tensor shaped like
    `token_indices` holding

        block_table[req_id[i], p // 32] * stride_factor + p % 32 + layer_id * 32

    or -1 when `p < 0`, `p // 32 >= block_table.size(1)`, or the block-table
    entry is negative. No argument is modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls it once per layer on the step's whole top-k buffer:
    # decode steps of up to 32 sequences x MTP 2 at MTP widths other than 2, and
    # every prefill step. `max_blocks` is the engine's block-table width.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one sequence, one token, a context longer than top-k",
            spec=dict(**_GLM, seq_rows=[1], kv_lens=[4096], max_blocks=512, layer_id=0),
        ),
        Cell(
            why="the full decode batch: 32 sequences x MTP 2, the last pool layer",
            spec=dict(
                **_GLM,
                seq_rows=[2] * 32,
                kv_lens=[300 + 251 * i for i in range(32)],
                max_blocks=512,
                layer_id=78,
            ),
        ),
        Cell(
            why="contexts shorter than top-k: most columns are -1 padding",
            spec=dict(**_GLM, seq_rows=[2, 2], kv_lens=[33, 1000], max_blocks=512, layer_id=5),
        ),
        Cell(
            why="prefill: 2048 rows of one request, causal windows",
            spec=dict(**_GLM, seq_rows=[2048], kv_lens=[2048], max_blocks=512, layer_id=40),
        ),
        Cell(
            why="positions past the table and unused pages: both come back -1",
            spec=dict(
                **_GLM,
                seq_rows=[1, 1],
                kv_lens=[5000, 64],
                max_blocks=64,
                layer_id=1,
                pad_block_table=True,
            ),
        ),
    )

    note = """
    `build(device_id, topk=, max_blocks=)` compiles the launcher; the target
    builds it on its first forward, when the block table's width is known, and
    passes it as `compiled`. Every row count runs the kernel, so there is no
    stock-op fallback. The target's MTP layer keeps the stock
    `convert_req_index_to_global`.

    `block_size` must be 32; the wrapper rejects any other loudly, as
    it does non-contiguous or non-int32 operands and a `token_indices` width
    that is not a multiple of 256.

    For the decode kernel that adds the layer offset itself, see
    `cute_dsl_req_index_to_base`.
    """

    def __call__(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
        *,
        compiled: CompiledReqIndex,
    ) -> torch.Tensor:
        return _kernel.convert_req_index_to_global_rawptr(
            req_id,
            block_table,
            token_indices,
            int(block_size),
            int(stride_factor),
            int(layer_id),
            compiled.fn,
        )

    def reference(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
        *,
        compiled: CompiledReqIndex,
    ) -> torch.Tensor:
        """The integer map in int64, so it cannot wrap; `compiled` is not read."""
        return pool_rows(
            req_id, block_table, token_indices, block_size, stride_factor, layer_id * block_size
        )

    def is_valid(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        stride_factor: int,
        layer_id: int,
        *,
        compiled: CompiledReqIndex,
    ) -> None:
        check_compiled(compiled, block_table, token_indices)
        check_int32_range(block_table, stride_factor, layer_id)


cute_dsl_req_index_to_global = CuteDslReqIndexToGlobal()
