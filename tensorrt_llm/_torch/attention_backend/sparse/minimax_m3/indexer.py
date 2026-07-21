# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA sparse-attention indexer.

Mirrors the DSA `Indexer` pattern: a small submodule owned by the
sparse-attention backend that runs the predictor pass and produces the
per-query selected KV block indices the main attention consumes.

The proxy attention calls `fmha_sm100` directly (like DSA's
`sparse_attn_indexer` calls `fp8_paged_mqa_logits`) rather than wrapping
it as an `Fmha` registry library:

  * Prefill runs `fmha_sm100_plan` + `fmha_sm100` eagerly in
    `output_maxscore` mode, reduces the per-index-head max score to
    KV-head granularity, and selects top-k blocks per query.
  * Decode runs through the CUDA-graph-safe in-tree driver
    (`decode_wrapper.dispatch`), which itself calls the same `fmha_sm100`
    kernel binaries with device-tensor launch arguments.

The selected block indices are returned as `[total_q, num_kv_heads,
topk]` int32 (ascending, -1 padded) and published on
`forward_args.sparse_prediction.sparse_attn_indices` by the owning
`sparse_attn_predict`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .common import (
    _MSA_REQUIRED_TOPK,
    build_kv_indices_and_lens,
    idx_cache_to_msa_paged,
    per_token_valid_blocks,
    require_msa_module,
    select_blocks_from_maxscore,
)

if TYPE_CHECKING:
    from .metadata import MiniMaxM3SparseConfig


def _proxy_max_score_direct(
    idx_q: torch.Tensor,
    idx_k_paged: torch.Tensor,
    *,
    qo_lens_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    qo_offset_cpu: Optional[torch.Tensor],
    kv_indices: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> torch.Tensor:
    """Direct `fmha_sm100` MQA proxy pass (no Fmha-lib wrapper).

    Follows MSA's two-call pattern: `fmha_sm100_plan` builds the plan with
    `output_maxscore=True` and `num_kv_heads=1` (MQA), and `fmha_sm100`
    runs with `output_o=False` so only the per-block max score is
    materialized. Returns `[num_index_heads, max_k_tiles, total_q]`
    float32.
    """
    fmha_sm100 = require_msa_module()

    if idx_q.dim() != 3:
        raise ValueError(
            "MsaIndexer expects idx_q with shape [total_q, num_index_heads, head_dim]; "
            f"got {tuple(idx_q.shape)}."
        )
    if idx_k_paged.dim() != 4 or idx_k_paged.shape[1] != 1:
        raise ValueError(
            "MsaIndexer expects MQA paged index-K [num_pages, 1, page_size, head_dim]; "
            f"got {tuple(idx_k_paged.shape)}."
        )

    page_size = int(idx_k_paged.shape[2])
    proxy_plan = fmha_sm100.fmha_sm100_plan(
        qo_lens_cpu,
        kv_lens_cpu,
        idx_q.shape[1],  # num_index_heads (= num_qo_heads for the MQA proxy)
        num_kv_heads=1,
        qo_offset=qo_offset_cpu,
        page_size=page_size,
        output_maxscore=True,
        causal=causal,
        num_kv_splits=1,
    )
    _, max_score = fmha_sm100.fmha_sm100(
        idx_q,
        idx_k_paged,
        idx_k_paged,  # v passthrough; proxy ignores V via output_o=False
        proxy_plan,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
        sm_scale=sm_scale,
    )
    return max_score


def _group_max_reduce(max_score: torch.Tensor, config: "MiniMaxM3SparseConfig") -> torch.Tensor:
    """Reduce per-index-head max score to per-KV-head granularity (amax)."""
    if config.num_index_heads % config.num_kv_heads != 0:
        raise ValueError(
            f"num_index_heads ({config.num_index_heads}) must be divisible by "
            f"num_kv_heads ({config.num_kv_heads}) for MSA group-max reduction."
        )
    group = config.num_index_heads // config.num_kv_heads
    if group > 1:
        return max_score.view(
            config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]
        ).amax(dim=1)
    return max_score


class MsaIndexer:
    """Predictor submodule: proxy MQA and top-k block selection.

    Owned by `MiniMaxM3MSATrtllmAttention` and stateless: the decode pass
    borrows the per-metadata-instance `M3DecodePlanner`.
    """

    def __init__(self, config: "MiniMaxM3SparseConfig"):
        self.config = config

    # prefill: eager, direct fmha_sm100

    def select_blocks_prefill(
        self,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        *,
        idx_sm_scale: float,
        qo_lens_cpu: torch.Tensor,
        kv_lens_cpu: torch.Tensor,
        qo_offset_cpu: Optional[torch.Tensor],
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[total_q, num_kv_heads, topk]` selected block indices."""
        config = self.config
        idx_k_paged = idx_cache_to_msa_paged(idx_k_cache)

        max_score = _proxy_max_score_direct(
            idx_q,
            idx_k_paged,
            qo_lens_cpu=qo_lens_cpu,
            kv_lens_cpu=kv_lens_cpu,
            qo_offset_cpu=qo_offset_cpu,
            kv_indices=kv_indices,
            sm_scale=idx_sm_scale,
            causal=True,
        )
        max_score_kv = _group_max_reduce(max_score, config)

        page_size = int(idx_k_paged.shape[2])
        n_valid_blocks = per_token_valid_blocks(
            qo_lens_cpu,
            kv_lens_cpu,
            qo_offset_cpu,
            causal=True,
            block_size=page_size,
        )
        if n_valid_blocks.numel() == 0 or int(n_valid_blocks.max().item()) <= 0:
            return torch.full(
                (idx_q.shape[0], config.num_kv_heads, _MSA_REQUIRED_TOPK),
                -1,
                dtype=torch.int32,
                device=idx_q.device,
            )
        return select_blocks_from_maxscore(
            max_score_kv,
            topk=_MSA_REQUIRED_TOPK,
            n_valid_blocks=n_valid_blocks,
            init_blocks=config.init_blocks,
            local_blocks=config.local_blocks,
        )

    # decode: CUDA-graph-safe in-tree driver

    def select_blocks_decode(
        self,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        metadata,
        *,
        idx_sm_scale: float,
        page_size: int,
    ) -> torch.Tensor:
        """Return `kv_block_indexes` via the graph-safe decode planner.

        `metadata` is the owning `MiniMaxM3MSATrtllmAttentionMetadata`;
        its per-instance `M3DecodePlanner` runs the same `fmha_sm100`
        proxy binary plus a device-side top-k, so this path is CUDA-graph
        capturable.
        """
        idx_k_paged = idx_cache_to_msa_paged(idx_k_cache)
        batch = int(idx_q.shape[0])
        m3_meta = metadata.m3_meta
        seq_lens = m3_meta.seq_lens.to(torch.int32)

        plans = metadata.m3_plans
        if plans is not None:
            kv_indices = plans.kv_indices
            kv_page_indptr = plans.kv_page_indptr
        else:
            # Eager-only fallback (focused tests / first warmup step).
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "MiniMax-M3 MSA decode reached the eager fallback during CUDA graph "
                    "capture: plan tables were not pre-staged by prepare()."
                )
            kv_indices, _ = build_kv_indices_and_lens(m3_meta, page_size)
            num_pages_cpu = (m3_meta.seq_lens_cpu.to(torch.long) + page_size - 1) // page_size
            kv_page_indptr = torch.zeros(batch + 1, dtype=torch.int32)
            kv_page_indptr[1:] = num_pages_cpu.to(torch.int32).cumsum(0)
            kv_page_indptr = kv_page_indptr.to(idx_q.device, non_blocking=True)

        planner = metadata.m3_decode_planner(batch)
        max_score = planner.proxy_max_score(
            idx_q,
            idx_k_paged,
            seq_lens=seq_lens,
            kv_page_indptr=kv_page_indptr,
            kv_indices=kv_indices,
            sm_scale=idx_sm_scale,
        )
        return planner.select_blocks(max_score, seq_lens=seq_lens)


__all__ = ["MsaIndexer"]
