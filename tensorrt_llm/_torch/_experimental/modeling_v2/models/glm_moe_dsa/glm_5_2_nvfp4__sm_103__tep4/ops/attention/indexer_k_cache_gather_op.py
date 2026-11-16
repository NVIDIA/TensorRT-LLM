# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather one prefill chunk's DSA indexer keys back out of the indexer K cache."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: GLM-5.2's indexer cache: 128 fp8 bytes plus one fp32 scale per token, 32
#: tokens per page. A page stores all 32 tokens' data first, then their scales.
_GLM = dict(head_dim=128, tokens_per_block=32)


class IndexerKCacheGatherOp(OpWrapper):
    """Read `num_tokens` cached indexer keys into two fresh tensors.

    Output row `i` is the `head_dim` bytes at logical flat byte offset
    `slot_mapping_fp8[k_token_start + i]` of `k_cache`, and its scale the four
    bytes at `slot_mapping_scale[k_token_start + i]`. "Logical" means the
    offset indexes `k_cache` as if it were contiguous: the kernel unravels it
    over `k_cache`'s shape and applies its real strides, so a layer view of an
    interleaved pool works.

    Returns `(data [num_tokens, head_dim] float8_e4m3fn, scale [num_tokens, 1]
    float32)`. `k_cache` is not modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM's prefill reads each chunk's keys once per layer: the cached prefix
    # plus the chunk's own new keys. The cells vary the chunk's key count, where
    # in the mapping the chunk starts, how the keys are spread over pages, and
    # whether the cache is one layer's strided view of a shared pool.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a short chunk: one page and a bit, from the start of the mapping",
            spec=dict(**_GLM, seq_lens=[40], k_token_start=0, num_tokens=40, pool_layers=1),
        ),
        Cell(
            why="prefill chunk: 2048 keys of one request",
            spec=dict(**_GLM, seq_lens=[2048], k_token_start=0, num_tokens=2048, pool_layers=1),
        ),
        Cell(
            why="a later chunk: the window starts mid-mapping, across two requests",
            spec=dict(
                **_GLM, seq_lens=[1000, 3000], k_token_start=1000, num_tokens=3000, pool_layers=1
            ),
        ),
        Cell(
            why="one layer's view of a layer-interleaved pool: non-contiguous dim 0",
            spec=dict(
                **_GLM, seq_lens=[700, 1500], k_token_start=0, num_tokens=2200, pool_layers=3
            ),
        ),
        Cell(
            why="a long cached prefix: 8192 keys",
            spec=dict(**_GLM, seq_lens=[8192], k_token_start=0, num_tokens=8192, pool_layers=1),
        ),
    )

    note = """
    The kernel moves 4 bytes per thread, so every slot offset must be a
    multiple of 4; a misaligned one faults loudly rather than reading wrong.

    `num_tokens = 0` returns two empty tensors before any check runs.

    The FP4 cache (`head_dim = 64`) is the same kernel with half the bytes per
    token. GLM does not use it and this entry does not certify it.
    """

    def __call__(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.indexer_k_cache_gather_op(
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            k_token_start,
            num_tokens,
            head_dim,
        )

    def reference(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A byte gather over the cache's logical (contiguous) order."""
        flat = k_cache.contiguous().view(torch.uint8).reshape(-1)
        window = slice(k_token_start, k_token_start + num_tokens)
        data_at = slot_mapping_fp8[window].long().unsqueeze(1)
        scale_at = slot_mapping_scale[window].long().unsqueeze(1)
        data = flat[data_at + torch.arange(head_dim, device=flat.device)]
        scale = flat[scale_at + torch.arange(4, device=flat.device)]
        return (
            data.view(torch.float8_e4m3fn),
            scale.contiguous().view(torch.float32).reshape(num_tokens, 1),
        )

    def is_valid(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> None:
        # The op checks only the upper end of the window. A negative start reads
        # the int64 words in front of the mapping and uses them as slots.
        assert k_token_start >= 0, f"k_token_start must be >= 0, got {k_token_start}"
        if num_tokens == 0:
            return
        # A negative slot makes the kernel skip that row, and the output is
        # torch.empty: the row comes back as whatever the allocator held.
        window = slice(k_token_start, k_token_start + num_tokens)
        assert bool((slot_mapping_fp8[window] >= 0).all()) and bool(
            (slot_mapping_scale[window] >= 0).all()
        ), "every slot in the window must be >= 0; a negative slot leaves its row uninitialized"


indexer_k_cache_gather_op = IndexerKCacheGatherOp()
