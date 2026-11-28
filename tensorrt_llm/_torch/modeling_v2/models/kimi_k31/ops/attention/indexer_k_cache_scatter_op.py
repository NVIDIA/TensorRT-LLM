# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer: write FP8 index keys and their scales into the paged index-key cache (CUDA)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def _spec(num_tokens, **overrides) -> dict:
    spec = dict(
        num_tokens=num_tokens,
        tokens_per_block=64,
        num_pages=64,
        page_stride_pad=0,
        slot_rows=None,
    )
    spec.update(overrides)
    return spec


def unravel_bytes(flat: torch.Tensor, shape) -> tuple[torch.Tensor, ...]:
    """Flat byte offsets of a [pages, P, 1, 132] cache to its four indices (logical shape)."""
    i3 = flat % shape[3]
    flat = flat // shape[3]
    i2 = flat % shape[2]
    flat = flat // shape[2]
    i1 = flat % shape[1]
    return flat // shape[1], i1, i2, i3


class IndexerKCacheScatterOp(OpWrapper):
    """For ``t < num_tokens``: the 128 bytes of ``k_fp8[t]`` go to flat byte offsets
    ``slot_mapping_fp8[t] + [0, 128)`` and the 4 bytes of ``k_scale[t]`` to
    ``slot_mapping_scale[t] + [0, 4)`` of ``k_cache`` ([pages, P, 1, 132] uint8). A flat offset is
    unravelled over the cache's logical shape and stored through its strides, so a page view with
    a padded page stride is addressed correctly. ``k_fp8`` is E4M3 [T, 128], ``k_scale`` a 4-byte
    dtype [T, 1] (the indexer passes the fp32 scales viewed as int32), the slot mappings int64
    [>= num_tokens]. Writes in place; returns None.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="one decode token", spec=_spec(1)),
        Cell(why="a DFlash2 w8 verify step of 4 requests: 32 tokens", spec=_spec(32)),
        Cell(why="a 3000-token prefill chunk over random pages", spec=_spec(3000)),
        Cell(why="32-token pages", spec=_spec(500, tokens_per_block=32)),
        Cell(why="128-token pages", spec=_spec(500, tokens_per_block=128)),
        Cell(
            why="a padded page stride: the V2 manager's per-layer view of a wider page",
            spec=_spec(700, page_stride_pad=256),
        ),
        Cell(
            why="num_tokens below the slot-mapping length: rows past it are left alone",
            spec=_spec(100, slot_rows=160),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300) and sm_107 (VR200).

    The K3.1 general target calls this once per DSA layer per step with the step's FP8 index keys
    (fused_cat_fp8 of the normalized wk projection), their fp32 scales viewed as int32, the
    layer's index-key cache view, the metadata's precomputed slot mappings (flat byte offsets of
    each token's key bytes and scale bytes within its page: the P rows of 128 key bytes come
    first, then the P scales) and the step's token count. Byte copies only: exact and
    deterministic; distinct tokens must map to distinct slots. Captures in a CUDA graph.
    """

    def __call__(
        self,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> None:
        torch.ops.trtllm.indexer_k_cache_scatter_op(
            k_fp8, k_scale, k_cache, slot_mapping_fp8, slot_mapping_scale, num_tokens
        )

    def reference(
        self,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> None:
        """Byte scatter through the logical shape, in place."""
        shape = k_cache.shape
        data = k_fp8[:num_tokens].contiguous().view(torch.uint8)
        width = data.shape[1]
        flat = slot_mapping_fp8[:num_tokens, None] + torch.arange(width, device=k_cache.device)
        k_cache[unravel_bytes(flat, shape)] = data
        scale = k_scale[:num_tokens].contiguous().view(torch.uint8).reshape(num_tokens, -1)
        flat = slot_mapping_scale[:num_tokens, None] + torch.arange(
            scale.shape[1], device=k_cache.device
        )
        k_cache[unravel_bytes(flat, shape)] = scale

    def is_valid(
        self,
        k_fp8: torch.Tensor,
        k_scale: torch.Tensor,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        num_tokens: int,
    ) -> None:
        # Offsets past the cache's logical size write into whatever follows it.
        logical = k_cache.shape[0] * k_cache.shape[1] * k_cache.shape[2] * k_cache.shape[3]
        assert int(slot_mapping_fp8[:num_tokens].max()) + k_fp8.shape[1] <= logical, (
            "a key slot past the cache"
        )
        assert int(slot_mapping_scale[:num_tokens].max()) + 4 <= logical, (
            "a scale slot past the cache"
        )
        assert slot_mapping_fp8.shape[0] >= num_tokens and k_fp8.shape[0] >= num_tokens, (
            "num_tokens exceeds the slot mappings or the keys"
        )


indexer_k_cache_scatter_op = IndexerKCacheScatterOp()
