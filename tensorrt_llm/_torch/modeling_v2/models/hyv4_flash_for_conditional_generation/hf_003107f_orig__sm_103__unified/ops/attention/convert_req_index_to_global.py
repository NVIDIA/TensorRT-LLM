# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Map a request-local KV position to its row in the primary latent pool.

Adapted from the GLM-5.2 target's `ops/attention/convert_req_index_to_global.py`
(same engine op, cells re-derived for this checkpoint). HY4-Flash hands the
sparse MLA entry its selection built from `position_ids` -- request-local
positions, ascending, `-1` past the document start -- while the trtllm-gen
sparse kernel reads `kvPageIdxPtr` as rows of the pool base it is given. This
entry is the conversion between the two.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: HY4-Flash's primary-pool geometry: pages of 32 tokens, 2048 selected keys on a
#: sparse layer and 512 on a windowed one.
_HY4 = dict(block_size=32, topk=2048)


class ConvertReqIndexToGlobal(OpWrapper):
    """Turn a layer's per-row key positions into rows of the primary KV pool.

    Entry `token_indices[i, j]` is a KV position `p` counted from the start of
    request `req_id[i]`'s own sequence. The op returns

        block_table[req_id[i], p // block_size] * stride_factor
            + layer_id * block_size + p % block_size

    which is the row of the pool viewed as `[-1, 576]` that holds that token's
    latent KV for this layer. A position that cannot be resolved comes back as
    -1: `p < 0` (the selection's padding), `p // block_size` at or past
    `block_table.size(1)`, or a negative block-table entry (an unused page).

    Returns a fresh contiguous int32 tensor shaped like `token_indices`; no
    argument is modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # HY4-Flash calls it once per layer on the step's whole selection buffer: one
    # row per query token, `topk` columns -- 2048 on a sparse layer, the 512-token
    # window on a windowed one. `pool_layers` sets `stride_factor = pool_layers *
    # block_size`, the page stride of a pool that interleaves every layer's pages;
    # `layer_id` picks the layer inside a page. One primary pool holds all 42 text
    # layers in decoder order.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one sequence, one token, a context longer than top-k",
            spec=dict(**_HY4, seq_rows=[1], kv_lens=[5120], pool_layers=42, layer_id=0),
        ),
        Cell(
            why="the full decode batch: 64 sequences, one token, the last pool layer",
            spec=dict(
                **_HY4,
                seq_rows=[1] * 64,
                kv_lens=[300 + 73 * i for i in range(64)],
                pool_layers=42,
                layer_id=41,
            ),
        ),
        Cell(
            why="windowed layers: 512 columns per row, a decode batch past the window",
            spec=dict(
                _HY4, topk=512, seq_rows=[1] * 8, kv_lens=[5120] * 8, pool_layers=42, layer_id=2
            ),
        ),
        Cell(
            why="contexts shorter than top-k: most columns are -1 padding",
            spec=dict(**_HY4, seq_rows=[2, 2], kv_lens=[33, 1000], pool_layers=42, layer_id=5),
        ),
        Cell(
            why="prefill chunk: 4096 rows of one request, causal windows",
            spec=dict(**_HY4, seq_rows=[4096], kv_lens=[4096], pool_layers=42, layer_id=13),
        ),
        Cell(
            why="padded block table and positions past it: both must come back -1",
            spec=dict(
                **_HY4,
                seq_rows=[1, 1],
                kv_lens=[5000, 64],
                pool_layers=42,
                layer_id=1,
                pad_block_table=True,
            ),
        ),
    )

    note = """
    The kernel launches one block row per `req_id` entry and writes
    `num_topk_tokens` columns at the output's row stride, which is
    `token_indices.size(1)`. Both counts are taken on trust; see `is_valid`.

    Arithmetic is int32 on the device: `block * stride_factor` past 2**31
    wraps. This checkpoint's pool is far below that; `is_valid` states the
    bound. The windowed layers' 512-column selection is the same call at a
    narrower `num_topk_tokens`; both widths are cells above.

    The grouped variant (`convert_req_index_to_global_grouped`, one launch for
    several layers) is a different op and is not covered here.
    """

    def __call__(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.convert_req_index_to_global(
            req_id,
            block_table,
            token_indices,
            block_size,
            num_topk_tokens,
            stride_factor,
            layer_id,
        )

    def reference(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> torch.Tensor:
        """The kernel's integer map, in int64 so it cannot wrap.

        `num_topk_tokens` is not read: the reference covers every column of
        `token_indices`, which `is_valid` requires to be the same count.
        """
        tok = token_indices.long()
        max_blocks = block_table.shape[1]
        block = torch.div(tok, block_size, rounding_mode="floor")
        in_block = tok - block * block_size + layer_id * block_size
        valid = (tok >= 0) & (block < max_blocks)
        rows = req_id.long().unsqueeze(1).expand_as(tok)
        base = block_table.long()[rows, block.clamp(0, max_blocks - 1)]
        valid &= base >= 0
        out = torch.where(valid, base * stride_factor + in_block, torch.full_like(tok, -1))
        return out.to(torch.int32)

    def is_valid(
        self,
        req_id: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        block_size: int,
        num_topk_tokens: int,
        stride_factor: int,
        layer_id: int,
    ) -> None:
        # The output's row stride is `token_indices.size(1)`, but the kernel
        # walks `num_topk_tokens` columns. Wider spills each row's surplus into
        # the next row and past the end of the output; narrower leaves the
        # trailing columns uninitialized. Neither raises.
        assert num_topk_tokens == token_indices.shape[1], (
            f"num_topk_tokens must equal token_indices.size(1); got {num_topk_tokens} "
            f"against a {tuple(token_indices.shape)} operand"
        )
        # The row count is `req_id.size(0)`, not `token_indices.size(0)`. A
        # longer `req_id` reads past `token_indices` and writes past the output;
        # a shorter one leaves the extra output rows uninitialized.
        assert req_id.shape[0] == token_indices.shape[0], (
            f"req_id must hold exactly one entry per row of token_indices; got "
            f"{req_id.shape[0]} for {token_indices.shape[0]} rows"
        )
        # int32 arithmetic on the device: a pool row past 2**31 wraps silently.
        if block_table.numel():
            top = int(block_table.max()) * stride_factor + (layer_id + 1) * block_size
            assert top < 2**31, f"pool row {top} does not fit int32; the kernel wraps it silently"


convert_req_index_to_global = ConvertReqIndexToGlobal()
