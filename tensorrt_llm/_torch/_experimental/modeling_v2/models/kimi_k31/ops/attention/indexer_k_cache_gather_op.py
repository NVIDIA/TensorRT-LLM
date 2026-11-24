# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA indexer: read a prefill chunk's FP8 index keys and scales out of the paged cache (CUDA)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .indexer_k_cache_scatter_op import unravel_bytes


def _spec(num_tokens, **overrides) -> dict:
    spec = dict(
        num_tokens=num_tokens,
        k_token_start=0,
        tokens_per_block=64,
        num_pages=96,
        page_stride_pad=0,
    )
    spec.update(overrides)
    return spec


class IndexerKCacheGatherOp(OpWrapper):
    """Returns ``(k_fp8, k_scale)``: E4M3 [num_tokens, head_dim] from the bytes at
    ``slot_mapping_fp8[k_token_start + t] + [0, head_dim)`` and fp32 [num_tokens, 1] from the 4
    bytes at ``slot_mapping_scale[k_token_start + t]`` of ``k_cache`` ([pages, P, 1, 132] uint8,
    offsets unravelled over its logical shape and read through its strides). Both newly
    allocated.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="a one-token chunk", spec=_spec(1)),
        Cell(why="a 2049-key chunk", spec=_spec(2049)),
        Cell(
            why="the second chunk of a prefill: start offset into the full-KV slot mapping",
            spec=_spec(1500, k_token_start=2500),
        ),
        Cell(why="an 8192-key cached prefix plus its chunk", spec=_spec(5000, num_pages=160)),
        Cell(why="32-token pages", spec=_spec(700, tokens_per_block=32)),
        Cell(why="128-token pages", spec=_spec(700, tokens_per_block=128, num_pages=32)),
        Cell(
            why="a padded page stride: the V2 manager's per-layer view of a wider page",
            spec=_spec(700, page_stride_pad=256),
        ),
    )
    note = """
    ARCHS lists what was driven: sm_103 (GB300). The sm_107 cells are driven on VR200 by a
    separate run and are not listed until then.

    The K3.1 general target calls this in the DSA indexer's chunked prefill, once per chunk with
    cached or several requests' keys: the layer's index-key cache view, the metadata's full-KV
    slot mappings (cached and new tokens of the chunk's requests), the chunk's first key and key
    count, and head_dim 128 (bytes per FP8 key). Byte copies only: exact and deterministic. The
    output is fed to fp8_mqa_logits as (k, k_scale.reshape(-1)).
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
            k_cache, slot_mapping_fp8, slot_mapping_scale, k_token_start, num_tokens, head_dim
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
        """Byte gather through the logical shape."""
        shape = k_cache.shape
        rows = slice(k_token_start, k_token_start + num_tokens)
        flat = slot_mapping_fp8[rows, None] + torch.arange(head_dim, device=k_cache.device)
        k_fp8 = k_cache[unravel_bytes(flat, shape)].view(torch.float8_e4m3fn)
        flat = slot_mapping_scale[rows, None] + torch.arange(4, device=k_cache.device)
        k_scale = k_cache[unravel_bytes(flat, shape)].contiguous().view(torch.float32)
        return k_fp8, k_scale.reshape(num_tokens, 1)

    def is_valid(
        self,
        k_cache: torch.Tensor,
        slot_mapping_fp8: torch.Tensor,
        slot_mapping_scale: torch.Tensor,
        k_token_start: int,
        num_tokens: int,
        head_dim: int,
    ) -> None:
        # Offsets past the cache's logical size read whatever follows it.
        logical = k_cache.shape[0] * k_cache.shape[1] * k_cache.shape[2] * k_cache.shape[3]
        rows = slice(k_token_start, k_token_start + num_tokens)
        assert int(slot_mapping_fp8[rows].max()) + head_dim <= logical, "a key slot past the cache"
        assert int(slot_mapping_scale[rows].max()) + 4 <= logical, "a scale slot past the cache"


indexer_k_cache_gather_op = IndexerKCacheGatherOp()
