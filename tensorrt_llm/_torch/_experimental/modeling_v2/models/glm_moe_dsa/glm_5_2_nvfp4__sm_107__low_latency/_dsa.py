# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-owned sparse MLA projection and prefill dispatch."""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSABackendForwardArgs
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, is_torch_compiling


def _get_short_seq_mha_threshold() -> int:
    threshold = os.environ.get("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", "0")
    try:
        return int(threshold)
    except ValueError as err:
        raise ValueError(
            f"TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD must be an integer, got {threshold!r}"
        ) from err


def forward_context_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    indexer_intermediates: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    if should_use_short_mha(self, attn_metadata, position_ids):
        return self.forward_context(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )
    return self.forward_absorption_context(
        q,
        compressed_kv,
        k_pe,
        attn_metadata,
        output,
        position_ids=position_ids,
        latent_cache=latent_cache,
        sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
    )


def forward_generation_sparse_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    latent_cache: Optional[torch.Tensor] = None,
    indexer_intermediates: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    return self.forward_absorption_generation(
        q,
        compressed_kv,
        k_pe,
        attn_metadata,
        output,
        position_ids=position_ids,
        latent_cache=latent_cache,
        sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
    )


def forward_sparse_attn(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    attn_output: list[torch.Tensor],
) -> None:
    """
    Forward pass for the MLA module with DSA (always in MQA mode).
    Writes result into output tensor in-place.

    Delegates to forward_dsa_proj (token-wise projections) followed by
    _forward_dsa_attn (batch-dependent attention dispatch).

    Args:
        position_ids (Optional[torch.IntTensor]): The position IDs.
        hidden_states (torch.Tensor): The hidden states.
        attn_metadata (AttentionMetadata): The attention metadata.
        attn_output: Algorithm-defined output tensors. DSA writes to the
            standard output at index zero.
    """
    output = attn_output[0]
    proj_outputs = forward_dsa_proj(self, position_ids, hidden_states, attn_metadata)
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    _forward_dsa_attn(
        self,
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        attn_metadata,
        output,
    )


def forward_sparse_attn_custom_op(
    self,
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    attn_output: list[torch.Tensor],
    latent_cache_gen: Optional[torch.Tensor],
) -> None:
    """Run DSA's two-stage custom ops for piecewise CUDA graph capture."""
    del latent_cache_gen
    from . import _dsa_custom_ops as custom_ops

    if isinstance(hidden_states, Fp4QuantizedTensor):
        proj_outputs = custom_ops.mla_dsa_proj(
            hidden_states.unquantized_hidden_states,
            position_ids,
            self.layer_idx_str,
            hidden_states.fp4_tensor,
            hidden_states.scaling_factor,
        )
    else:
        proj_outputs = custom_ops.mla_dsa_proj(hidden_states, position_ids, self.layer_idx_str)
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    custom_ops.maybe_bcg_mla_dsa_attn_inplace(
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        self.layer_idx_str,
        attn_output[0],
    )


def forward_dsa_proj(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable Op 1).

    Runs kv_a_proj, layernorms, q_b_proj, and conditionally
    indexer.pre_indexer_proj().

    IMPORTANT: This method must NOT slice tensors by num_tokens or
    access batch-specific metadata, so that all operations are
    unconditionally straight-line for CUDA graph capture.  Slicing
    to num_tokens happens in _forward_dsa_attn (Op 2, outside graph).

    Returns [q, compressed_kv, k_pe, latent_cache] when short-MHA
    handles all tokens (eager only), or
    [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
    weights, q_scale] when the indexer runs.  q_scale is unused on the
    FP8 path but always present so CUDA graph capture sees a stable
    9-tensor shape regardless of indexer dtype.
    """
    assert self.mqa is not None, "DSA is only supported in MQA mode"

    q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
        [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
    )

    q_pair, compressed_kv = maybe_execute_in_parallel(
        lambda: self._q_a_layernorm_maybe_fused(q, return_norm_out=True),
        lambda: self.kv_a_layernorm(compressed_kv),
        self.ln_events[0],
        self.ln_events[1],
        self.aux_stream,
    )
    q, qr = q_pair
    latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

    q = self.q_b_proj(q)

    use_short_mha_for_ctx = should_use_short_mha(self, attn_metadata, position_ids)

    # Skip the indexer when the short MHA path handles all context
    # tokens and there are no generation tokens.
    if use_short_mha_for_ctx and attn_metadata.num_generations == 0:
        return [q, compressed_kv, k_pe, latent_cache]

    # DSA "shared" layer: no indexer projection; the backend predictor
    # reuses the preceding full layer's top-k.
    if self.mqa.indexer is None:
        return [q, compressed_kv, k_pe, latent_cache]

    # pre_indexer_proj is the CUDA-graph-safe portion: pure token-wise
    # compute (cublas_mm, rope, FP8 quantize, weight scaling) with no
    # access to batch-specific metadata or the k cache. Query scales are
    # already folded into weights; the fifth output keeps the graph contract.
    q_fp8, k_fp8, k_scale, weights, q_scale = self.mqa.indexer.pre_indexer_proj(
        qr, hidden_states, position_ids
    )

    return [
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        q_scale,
    ]


def _forward_dsa_attn(
    self,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention for DSA MLA (Op 2, not graph-captured).

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale]
    when the indexer ran in Op 1, or [] when short-MHA handled all tokens.

    All num_tokens slicing happens here (not in Op 1) because
    num_tokens comes from batch-specific metadata and must not be
    baked into CUDA graph capture.
    """
    num_contexts = attn_metadata.num_contexts
    num_generations = attn_metadata.num_generations
    num_ctx_tokens = attn_metadata.num_ctx_tokens
    num_tokens = attn_metadata.num_tokens

    # Slice Op 1 outputs to actual num_tokens (Op 1 operates on the
    # full padded tensor for CUDA graph compatibility).
    q = q[:num_tokens, ...]
    compressed_kv = compressed_kv[:num_tokens, ...]
    k_pe = k_pe[:num_tokens, ...]
    latent_cache = latent_cache[:num_tokens, ...]
    if position_ids is not None:
        position_ids = position_ids[..., :num_tokens]

    use_short_mha_for_ctx = num_contexts > 0 and should_use_short_mha(
        self, attn_metadata, position_ids
    )

    if not (use_short_mha_for_ctx and num_generations == 0) and self.mqa.indexer is not None:
        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        # Slice indexer intermediates to actual num_tokens (they were
        # computed on the full padded tensor in Op 1).
        q_fp8 = q_fp8[:num_tokens, ...]
        k_fp8 = k_fp8[:num_tokens, ...]
        k_scale = k_scale[:num_tokens, ...]
        weights = weights[:num_tokens, ...]
        q_scale = q_scale[:num_tokens, ...]
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights, q_scale]
        # Update the full mixed batch once before context/generation prediction
        # splits. Context chunk gathers must observe the newly appended keys.
        self.mqa.indexer._update_k_cache(k_fp8, k_scale, attn_metadata)
    else:
        indexer_intermediates = None

    assert output is not None, "output must be provided"

    if num_contexts > 0:
        q_ctx = q[:num_ctx_tokens, ...]
        compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
        k_pe_ctx = k_pe[:num_ctx_tokens, ...]
        latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
        ctx_position_ids = position_ids[..., :num_ctx_tokens] if position_ids is not None else None
        if self.apply_rotary_emb:
            assert ctx_position_ids is not None
            k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, ctx_position_ids)

        forward_context_sparse_attn(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            output[:num_ctx_tokens, :],
            latent_cache_ctx,
            position_ids=ctx_position_ids,
            indexer_intermediates=indexer_intermediates,
        )

    if num_generations > 0:
        q_gen = q[num_ctx_tokens:, ...]
        compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
        k_pe_gen = k_pe[num_ctx_tokens:, ...]
        latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
        gen_position_ids = (
            position_ids[..., num_ctx_tokens:num_tokens] if position_ids is not None else None
        )
        if self.apply_rotary_emb:
            assert gen_position_ids is not None
            k_pe_gen = self.apply_rope(q_gen, k_pe_gen, gen_position_ids)

        forward_generation_sparse_attn(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            output[num_ctx_tokens:num_tokens, :],
            latent_cache=latent_cache_gen,
            position_ids=gen_position_ids,
            indexer_intermediates=indexer_intermediates,
        )

    # Join the prev_topk copy forked in sparse_attn_indexer; CUDA graph
    # capture requires the join within the same layer's forward.
    if self.mqa.indexer is not None:
        self.mqa.indexer.maybe_join_prev_topk_copy()


def should_use_short_mha(
    self, attn_metadata: AttentionMetadata, position_ids: Optional[torch.Tensor]
) -> bool:
    """Check if the short-seq MHA optimization should be used for context.

    Uses max_ctx_kv_len (max total KV length per context sequence,
    including cached tokens) when available, to correctly account for
    chunked context where the full attention span exceeds the threshold
    even if the new token count is small.  Falls back to num_ctx_tokens
    (total new context tokens) when max_ctx_kv_len is not set.

    Disabled under torch compile so that the split DSA custom ops
    (mla_dsa_proj / mla_dsa_attn_inplace) have unconditionally
    straight-line control flow for CUDA graph capture.
    """
    if is_torch_compiling():
        return False
    if not (
        self.short_seq_mha_threshold > 0
        and self.kv_cache_dtype != "fp8_ds_mla"
        and not self.apply_rotary_emb
        and self.mapping.cp_size == 1
        and position_ids is not None
    ):
        return False
    effective_len = getattr(attn_metadata, "max_ctx_kv_len", attn_metadata.num_ctx_tokens)
    return effective_len <= self.short_seq_mha_threshold
