# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared building blocks for the MiniMax-M3 sparse attention backends.

Both the Triton reference and the MSA (fmha_sm100) path share these
backend-neutral pieces: the lowered parameter and per-rank kernel config
bundles, block-priority sentinels, KV-slot writers, and the paged-cache
slot mapping builder. MSA-only helpers live in :mod:`.msa_utils`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import torch

from ..params import SparseMetadataParams, SparseParams

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping

# Sentinel scores that force init and local blocks into the top-k regardless
# of their computed score. Init outranks local.
_INIT_SCORE = 1e30
_LOCAL_SCORE = 1e29


@dataclass(frozen=True)
class MiniMaxM3SparseParams(SparseParams):
    """Lowered runtime parameters for the MiniMax-M3 sparse backend."""

    algorithm: Literal["minimax_m3"] = field(init=False, default="minimax_m3")
    num_index_heads: int = 4
    sparse_index_dim: int = 128
    block_size: int = 128
    topk: int = 16
    init_blocks: int = 0
    local_blocks: int = 1
    score_type: str = "max"
    disable_index_value: bool = True
    implementation: Literal["triton", "msa"] = "triton"
    indexer_kv_dtype: Literal["bf16", "fp8"] = "bf16"

    @property
    def indices_block_size(self) -> int:
        """Block granularity of the selected sparse indices.

        Read by the shared TrtllmAttention forward when publishing the
        sparse prediction. It equals the per-block scoring size.
        """
        return self.block_size


@dataclass(frozen=True)
class MiniMaxM3SparseMetadataParams(SparseMetadataParams):
    """Metadata-facing MiniMax-M3 sparse geometry."""

    global_num_q_heads: int = 0
    global_num_kv_heads: int = 0
    num_index_heads: int = 4
    topk: int = 16

    def sharded_head_counts(self, mapping: Optional["Mapping"] = None) -> Tuple[int, int]:
        """Return per-rank (num_q_heads, num_kv_heads) for mapping.

        Matches the model's attention sharding: no split under attention data
        parallelism, otherwise split by tp_size.
        """
        if mapping is not None and not getattr(mapping, "enable_attention_dp", False):
            tp_size = int(getattr(mapping, "tp_size", 1) or 1)
        else:
            tp_size = 1

        def _shard(num_heads: int) -> int:
            return (int(num_heads) + tp_size - 1) // tp_size

        return _shard(self.global_num_q_heads), _shard(self.global_num_kv_heads)


@dataclass(frozen=True)
class MiniMaxM3SparseConfig:
    """Per-rank kernel parameter bundle for MiniMax-M3 sparse attention.

    This is **not** a user-facing config (use
    :class:`tensorrt_llm.llmapi.llm_args.MiniMaxM3SparseAttentionConfig`
    for that). It is the layer-invariant, post-TP-shard parameter bundle
    that backend kernels and reference helpers consume. The user knobs
    come from :class:`MiniMaxM3SparseParams`; ``num_q_heads`` /
    ``num_kv_heads`` / ``head_dim`` come from the per-rank model
    geometry and must be supplied by the caller (typically via
    :meth:`from_sparse_params`).
    """

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_index_heads: int
    sparse_index_dim: int
    block_size: int
    topk: int
    init_blocks: int = 0
    local_blocks: int = 1
    score_type: str = "max"

    def __post_init__(self) -> None:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        if self.num_index_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_index_heads ({self.num_index_heads}) must be divisible "
                f"by num_kv_heads ({self.num_kv_heads})"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.topk <= 0:
            raise ValueError(f"topk must be > 0, got {self.topk}")
        if self.init_blocks < 0:
            raise ValueError(f"init_blocks must be >= 0, got {self.init_blocks}")
        if self.local_blocks < 0:
            raise ValueError(f"local_blocks must be >= 0, got {self.local_blocks}")
        if self.score_type != "max":
            # SGLang exposes only "max" today and that is what the MiniMax-M3
            # checkpoint config specifies. Reject anything else explicitly so
            # a config drift surfaces immediately.
            raise ValueError(
                f"score_type={self.score_type!r} is not supported "
                "(only 'max' matches the SGLang reference)"
            )

    @classmethod
    def from_sparse_params(
        cls,
        sparse_params: "MiniMaxM3SparseParams",
        *,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> "MiniMaxM3SparseConfig":
        """Build a kernel param bundle from lowered ``MiniMaxM3SparseParams``
        and the per-rank model geometry.
        """
        return cls(
            num_q_heads=int(num_q_heads),
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_dim),
            num_index_heads=int(sparse_params.num_index_heads),
            sparse_index_dim=int(sparse_params.sparse_index_dim),
            block_size=int(sparse_params.block_size),
            topk=int(sparse_params.topk),
            init_blocks=int(sparse_params.init_blocks),
            local_blocks=int(sparse_params.local_blocks),
            score_type=str(sparse_params.score_type),
        )


def write_kv_slots(
    cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    values: torch.Tensor,
    *,
    layout: Literal["NHD", "HND"] = "NHD",
) -> None:
    """Write per-token values into a K, V, or index-K cache at given slots.

    Handles a 3-D flat-slot cache and a 4-D paged view. `layout` sets the paged
    axis order: "NHD" is [num_pages, tokens_per_block, num_heads, channel],
    "HND" is [num_pages, num_heads, tokens_per_block, channel]. The paged view
    is non-contiguous, so the slot id is split into (page, within) and written
    by multi-dim assignment. `values` is always [num_tokens, num_heads, channel].
    """
    with torch.no_grad():
        if cache.ndim >= 4:
            token_axis = 2 if layout == "HND" else 1
            tokens_per_block = int(cache.shape[token_axis])
            out_long = out_cache_loc.to(torch.long)
            page = out_long // tokens_per_block
            within = out_long % tokens_per_block
            if layout == "HND":
                # Advanced indices on dims 0 and 2 broadcast to [num_tokens] and
                # move front, giving a [num_tokens, num_heads, channel] target.
                cache[page, :, within, :] = values.to(cache.dtype)
            else:
                cache[page, within] = values.to(cache.dtype)
        else:
            cache.index_copy_(0, out_cache_loc.to(torch.long), values.to(cache.dtype))


def build_paged_kv_slot_mapping(
    *,
    kv_cache_manager,
    request_ids,
    qo_lens_cpu: torch.Tensor,
    qo_offset_cpu: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the backend-neutral paged-cache slot mapping.

    Returns (req_to_token, slot_ids, out_cache_loc), derived only from the paged
    KV cache manager and the per-request query geometry, with no dependency on
    any backend-specific metadata.

    req_to_token is the [batch, max_kv_len] int32 map from (request, position)
    to a global slot id, expanded from get_block_ids_per_seq with
    tokens_per_block as block_id * tokens_per_block + offset_within_block.
    slot_ids is the [batch] identity row index into req_to_token. out_cache_loc
    lists the per-new-token slot ids in flattened query order: request b
    contributes positions qo_offset[b] through qo_offset[b] + qo_lens[b] - 1.
    That one formula covers prefill (qo_offset is the prefix length) and decode
    (qo_offset is kv_len - 1 with qo_len 1).

    The req_to_token reads that build out_cache_loc sync the host, so call this
    only from prepare(), never from the forward path.
    """
    tokens_per_block = int(kv_cache_manager.tokens_per_block)
    # block_ids_per_seq is a [batch, max_blocks_per_seq] tensor; row b holds the
    # block ids assigned to request_ids[b] in order.
    block_ids = kv_cache_manager.get_block_ids_per_seq(list(request_ids))
    batch = int(qo_lens_cpu.shape[0])
    max_blocks = int(block_ids.shape[1])
    max_kv_len = max_blocks * tokens_per_block

    # Expand block ids -> per-token slot ids.
    block_ids_dev = block_ids.to(device).to(torch.int64)
    within_block = torch.arange(tokens_per_block, device=device, dtype=torch.int64)
    # Outer product per batch entry: [batch, max_blocks, tokens_per_block]
    slot_grid = block_ids_dev.unsqueeze(-1) * tokens_per_block + within_block
    req_to_token = slot_grid.reshape(batch, max_kv_len).to(torch.int32)
    slot_ids = torch.arange(batch, device=device, dtype=torch.int32)

    # out_cache_loc: per-new-token slot ids, in flattened query-token order.
    req_to_token_cpu = req_to_token.to("cpu")
    qo_lens_list = qo_lens_cpu.to(torch.long).tolist()
    qo_offset_list = qo_offset_cpu.to(torch.long).tolist()
    out_cache_loc_list: List[int] = []
    for b in range(batch):
        start = int(qo_offset_list[b])
        for offset in range(int(qo_lens_list[b])):
            out_cache_loc_list.append(int(req_to_token_cpu[b, start + offset].item()))
    out_cache_loc = torch.tensor(out_cache_loc_list, dtype=torch.int32, device=device)
    return req_to_token, slot_ids, out_cache_loc


__all__ = [
    "MiniMaxM3SparseConfig",
    "MiniMaxM3SparseMetadataParams",
    "MiniMaxM3SparseParams",
    "build_paged_kv_slot_mapping",
    "write_kv_slots",
]
