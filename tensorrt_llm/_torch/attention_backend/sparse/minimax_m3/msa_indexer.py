# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 MSA sparse-attention indexer.

Mirrors the DSA indexer pattern: a submodule owned by the sparse backend
that runs the predictor pass and returns the per-query selected KV block
indices the main attention consumes. It calls fmha_sm100 directly in
output_maxscore mode, reduces the per-index-head max score to KV-head
granularity, and selects the top-k blocks per query.

Results are [total_q, num_kv_heads, topk] int32, ascending with -1 padding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from .msa_utils import (
    MSA_REQUIRED_TOPK,
    msa_kernel_choice,
    per_token_valid_blocks,
    require_msa_module,
    select_blocks_from_maxscore,
)

if TYPE_CHECKING:
    from .common import MiniMaxM3SparseConfig


def _cutedsl_score_runner():
    """Return the CuTe DSL indexer scoring runner, or None if unavailable.

    The CuTe DSL ops are registered only when the nvidia-cutlass-dsl package is
    importable, so this stays a soft dependency.
    """
    try:
        from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
    except ImportError:
        return None
    return getattr(cute_dsl_custom_ops, "CuteDSLMiniMaxM3IndexDecodeScoreRunner", None)


def _cutedsl_score(
    idx_q: torch.Tensor,
    idx_k_paged: torch.Tensor,
    max_score: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens_cuda: torch.Tensor,
    decode_query_len: int,
) -> bool:
    """Try to fill `max_score` with the CuTe DSL scorer; report whether it ran.

    `max_score` is the [num_index_heads, max_k_tiles, total_q] buffer the block
    selector consumes. The kernel writes [head, token, block], so it is handed
    the transposed view: same backing store, no copy, and the stores end up
    coalesced across tokens rather than strided by max_k_tiles.

    The buffer is deliberately not pre-filled with -inf. The kernel writes
    blocks [0, ceil(seq_len / page_size)) for every token of a request, and the
    selector reads only [0, n_valid_blocks[token])), which is bounded by that
    same count for every token including the shorter ones in a multi-token
    speculative step. So every entry the selector reads has just been written.
    """
    runner = _cutedsl_score_runner()
    if runner is None:
        return False

    total_q, num_index_heads, head_dim = idx_q.shape
    page_size = int(idx_k_paged.shape[2])
    if not runner.is_supported(
        q_dtype=idx_q.dtype,
        num_heads=num_index_heads,
        head_dim=head_dim,
        page_size=page_size,
        max_decode_query_len=decode_query_len,
    ):
        return False
    if idx_k_paged.dtype != idx_q.dtype or max_score.shape[2] != total_q:
        return False

    # The kernel wants MQA index-K as [num_pages, page_size, head_dim]; the
    # squeeze is zero-copy and keeps the pool's real per-page stride, which the
    # TMA descriptor reads at runtime.
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q,
        idx_k_paged.squeeze(1),
        block_table,
        seq_lens_cuda,
        max_score.transpose(1, 2),
        decode_query_len,
    )
    return True


def _proxy_max_score(
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
    """Run the fmha_sm100 MQA proxy pass and return the per-block max score.

    Follows MSA's two-call pattern: fmha_sm100_plan builds the plan with
    output_maxscore and num_kv_heads 1, then fmha_sm100 runs with output_o
    disabled so only the per-block max score is produced. Returns
    [num_index_heads, max_k_tiles, total_q] float32.
    """
    fmha_sm100 = require_msa_module()

    if idx_q.dim() != 3:
        raise ValueError(
            "MsaIndexer expects idx_q [total_q, num_index_heads, head_dim]; "
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
        idx_q.shape[1],
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
        idx_k_paged,
        proxy_plan,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
        sm_scale=sm_scale,
    )
    return max_score


def _group_max_reduce(
    max_score: torch.Tensor,
    config: "MiniMaxM3SparseConfig",
) -> torch.Tensor:
    """Reduce per-index-head max score to per-KV-head granularity by amax.

    Index heads are assumed to be grouped contiguously per KV head, so head h
    maps to KV group h // group.
    """
    group, rem = divmod(config.num_index_heads, config.num_kv_heads)
    if rem != 0:
        raise ValueError(
            "num_index_heads must be divisible by num_kv_heads for group max "
            f"reduce; got num_index_heads={config.num_index_heads}, "
            f"num_kv_heads={config.num_kv_heads}."
        )
    if group > 1:
        return max_score.view(
            config.num_kv_heads, group, max_score.shape[1], max_score.shape[2]
        ).amax(dim=1)
    return max_score


class MsaIndexer:
    """Predictor submodule: proxy MQA scoring and top-k block selection.

    Owned by the MSA attention layer. Stateless in eager mode: it reads the
    per-forward page table and lengths from the attention metadata and calls
    the kernel directly.
    """

    def __init__(self, config: "MiniMaxM3SparseConfig"):
        self.config = config

    def select_blocks(
        self,
        idx_q: torch.Tensor,
        idx_k_paged: torch.Tensor,
        *,
        idx_sm_scale: float,
        kv_indices: torch.Tensor,
        qo_lens_cpu: Optional[torch.Tensor] = None,
        kv_lens_cpu: Optional[torch.Tensor] = None,
        qo_offset_cpu: Optional[torch.Tensor] = None,
        proxy_plan: Optional[tuple] = None,
        max_score: Optional[torch.Tensor] = None,
        n_valid_blocks: Optional[torch.Tensor] = None,
        head_major_output: bool = False,
        block_table: Optional[torch.Tensor] = None,
        seq_lens_cuda: Optional[torch.Tensor] = None,
        decode_query_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Return [total_q, num_kv_heads, topk] selected block indices.

        Plan/run split, mirroring the sparse GQA. Both production paths pass a
        prebuilt `proxy_plan` and a precomputed device `n_valid_blocks` (decode
        from the graph-safe scratch, eager from the step-level device buffer);
        decode additionally runs into the preallocated `max_score` buffer inside
        the captured region.

        Pure-decode steps with a uniform per-request query length additionally
        pass `block_table`, `seq_lens_cuda` and `decode_query_len`, which lets
        the dedicated CuTe DSL scorer replace the fmha_sm100 proxy pass when
        TLLM_M3_INDEXER_SCORE selects it.
        """
        config = self.config

        scored = False
        if (
            msa_kernel_choice("TLLM_M3_INDEXER_SCORE") == "cutedsl"
            and max_score is not None
            and block_table is not None
            and seq_lens_cuda is not None
            and decode_query_len is not None
        ):
            # Like the fmha_sm100 proxy, whose max_score is read off the MMA
            # accumulator before the softmax scale, the CuTe DSL scorer emits
            # raw Q.K rather than idx_sm_scale * Q.K. Block ranking, and the
            # +inf forcing of the init/local blocks in
            # select_blocks_from_maxscore, are both invariant under a positive
            # scale, so neither depends on the omission.
            scored = _cutedsl_score(
                idx_q,
                idx_k_paged,
                max_score,
                block_table=block_table,
                seq_lens_cuda=seq_lens_cuda,
                decode_query_len=decode_query_len,
            )

        if not scored:
            if proxy_plan is None:
                max_score = _proxy_max_score(
                    idx_q,
                    idx_k_paged,
                    qo_lens_cpu=qo_lens_cpu,
                    kv_lens_cpu=kv_lens_cpu,
                    qo_offset_cpu=qo_offset_cpu,
                    kv_indices=kv_indices,
                    sm_scale=idx_sm_scale,
                    causal=True,
                )
            else:
                fmha_sm100 = require_msa_module()
                _, max_score = fmha_sm100.fmha_sm100(
                    idx_q,
                    idx_k_paged,
                    idx_k_paged,
                    proxy_plan,
                    kv_indices=kv_indices,
                    output_o=False,
                    output_maxscore=True,
                    max_score=max_score,
                    sm_scale=idx_sm_scale,
                )

        max_score_kv = _group_max_reduce(max_score, config)

        if n_valid_blocks is None:
            n_valid_blocks = per_token_valid_blocks(
                qo_lens_cpu,
                kv_lens_cpu,
                qo_offset_cpu,
                causal=True,
                block_size=int(idx_k_paged.shape[2]),
            )
            # Empty-selection guard. n_valid_blocks is a host tensor on
            # this path, so the .item() read does not sync the device.
            if n_valid_blocks.numel() == 0 or int(n_valid_blocks.max().item()) <= 0:
                output_shape = (
                    (config.num_kv_heads, idx_q.shape[0], MSA_REQUIRED_TOPK)
                    if head_major_output
                    else (idx_q.shape[0], config.num_kv_heads, MSA_REQUIRED_TOPK)
                )
                output = torch.full(
                    output_shape,
                    -1,
                    dtype=torch.int32,
                    device=idx_q.device,
                )
                return output.permute(1, 0, 2) if head_major_output else output
        return select_blocks_from_maxscore(
            max_score_kv,
            topk=MSA_REQUIRED_TOPK,
            n_valid_blocks=n_valid_blocks,
            init_blocks=config.init_blocks,
            local_blocks=config.local_blocks,
            head_major_output=head_major_output,
        )


__all__ = ["MsaIndexer"]
