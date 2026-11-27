# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the glm-5.2-nvfp4 / sm_103 / tep4 Targets pass their ops about the step: the `Phase`
that maps the step's rows onto the compiled cells, and the batch state the context attention
reads off the prepared metadata (`build_step_args`)."""

from dataclasses import dataclass

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


@dataclass(frozen=True)
class Phase:
    """How a Target's rows map onto the compiled cells; a constant of the Target (and of its
    draft call). `wide`: rows above 64 are an MTP3 / MTP5 decode step's, on the MTP decode
    cells (`projection.wide_cell`) and the wide variants that keep the decode numerics. `up`:
    rows up to 64 without a cell of their own run on the next cell up at their exact rows -- on
    an MTP3 / MTP5 decode step (`"decode"`, `TRTLLM_GLM52_DEC_CELL_UP`) or an eager draft step
    (`"draft"`, which also lifts the calls that otherwise keep cuBLAS there); None keeps the
    stock op for such a count (`projection.cell_up`)."""

    wide: bool = False
    up: str | None = None


PREFILL = Phase()
DRAFT = Phase(up="draft")


def build_step_args(md: TrtllmAttentionMetadata) -> dict:
    """Project the prepared metadata onto the batch state the context attention reads: the
    rope/KV append's per-sequence offsets and pool addressing, the prefill kernel's pool and
    per-row KV windows."""
    return dict(
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        ctx_cached_token_indptr=md.ctx_cached_token_indptr,
        ctx_kv_indptr=md.ctx_kv_indptr,
        max_ctx_seq_len=md.max_ctx_seq_len,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        tokens_per_block=md.tokens_per_block,
        max_seq_len=md.max_seq_len,
        cu_seqlen_ks=md.cu_seqlen_ks,
        cu_seqlen_ke=md.cu_seqlen_ke,
        primary_pool=md.kv_cache_manager.get_unique_primary_pool(),
        # The prefill rope kernel's and the dense-causal FMHA's per-request lengths.
        ctx_uncached_token_indptr=md.ctx_uncached_token_indptr,
        # Their host twins, which the engine fills first and copies to the device: the
        # prefill rope kernel's int32 operands are staged from these.
        host_ctx_uncached_token_indptr=md.host_ctx_uncached_token_indptr,
        host_ctx_cached_token_indptr=md.host_ctx_cached_token_indptr,
        num_ctx_cached_tokens=md.num_ctx_cached_tokens,
        seq_lens=md.seq_lens,
        seq_lens_cuda=md.seq_lens_cuda,
    )
