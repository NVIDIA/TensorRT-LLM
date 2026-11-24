# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA custom ops used by piecewise CUDA graph capture.

Flat MLA projection and metadata dispatch."""

from __future__ import annotations

from typing import List, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph
from tensorrt_llm._torch.utils import Fp4QuantizedTensor

from ....ops.attention.dsa_params import DSABackendForwardArgs

# ====================================================================================================================
# Piecewise-CUDA-graph split ops of the DSA MLA
# ====================================================================================================================


def _extract_extra_attrs(layer_idx: str):
    from tensorrt_llm._torch.attention.attention import extract_extra_attrs

    return extract_extra_attrs(layer_idx, "mla")


@torch.library.custom_op("trtllm::k31_nvfp4_mla_dsa_proj", mutates_args=())
def mla_dsa_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    hidden_states_fp4: Optional[torch.Tensor] = None,
    hidden_states_sf: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable).

    Runs kv_a_proj, layernorms, q_b_proj, and indexer.pre_indexer_proj
    (FP8 quantize, weight scaling).  Does NOT update the indexer k cache —
    that happens in Op 2 (mla_dsa_attn_inplace) because the scatter kernel
    accesses batch-specific metadata.

    Returns [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
    weights, q_scale] ([q, compressed_kv, k_pe, latent_cache] for a
    shared-indexer layer); the result length is fixed, keeping control flow
    straight-line for CUDA graph capture. The trailing q_scale is unused by
    the FP8 logits.
    """
    metadata, mla_layer = _extract_extra_attrs(layer_idx)
    if hidden_states_fp4 is not None or hidden_states_sf is not None:
        assert hidden_states_fp4 is not None and hidden_states_sf is not None, (
            "hidden_states_fp4 and hidden_states_sf must be passed together"
        )
        hidden_states = Fp4QuantizedTensor(
            fp4_tensor=hidden_states_fp4,
            scaling_factor=hidden_states_sf,
            unquantized_hidden_states=hidden_states,
        )
    return forward_dsa_proj(mla_layer, position_ids, hidden_states, metadata)


@mla_dsa_proj.register_fake
def _mla_dsa_proj_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    hidden_states_fp4: Optional[torch.Tensor] = None,
    hidden_states_sf: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    # 9 tensors: 4 attention inputs + 5 indexer intermediates.
    metadata, mla_layer = _extract_extra_attrs(layer_idx)
    num_tokens = hidden_states.shape[0]
    indexer = mla_layer.mqa.indexer
    q = hidden_states.new_empty([num_tokens, mla_layer.num_heads_tp * mla_layer.qk_head_dim])
    compressed_kv = hidden_states.new_empty([num_tokens, mla_layer.kv_lora_rank])
    k_pe = hidden_states.new_empty([num_tokens, mla_layer.qk_rope_head_dim])
    latent_cache = hidden_states.new_empty(
        [num_tokens, mla_layer.kv_lora_rank + mla_layer.qk_rope_head_dim]
    )
    if indexer is None:
        # DSA "shared" layer: no indexer, mirror forward_dsa_proj's early
        # return of only the 4 base tensors (no indexer intermediates).
        return [q, compressed_kv, k_pe, latent_cache]
    if getattr(indexer, "use_fp4", False):
        q_fp4 = hidden_states.new_empty(
            [num_tokens, indexer.n_heads, indexer.head_dim // 2], dtype=torch.int8
        )
        k_fp4 = hidden_states.new_empty([num_tokens, indexer.head_dim // 2], dtype=torch.int8)
        k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.int32)
        q_scale = hidden_states.new_empty([num_tokens, indexer.n_heads, 1], dtype=torch.int32)
        weights = hidden_states.new_empty([num_tokens, indexer.n_heads], dtype=torch.float32)
        return [q, compressed_kv, k_pe, latent_cache, q_fp4, k_fp4, k_scale, weights, q_scale]
    # Indexer intermediates: q_fp8, k_fp8, k_scale, weights, q_scale (FP8 index keys:
    # one float32 scale per token and per (token, head)).
    q_fp8 = hidden_states.new_empty(
        [num_tokens, indexer.n_heads, indexer.head_dim], dtype=torch.float8_e4m3fn
    )
    k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim], dtype=torch.float8_e4m3fn)
    k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.float32)
    q_scale = hidden_states.new_empty([num_tokens, indexer.n_heads, 1], dtype=torch.float32)
    weights = hidden_states.new_empty([num_tokens, indexer.n_heads], dtype=torch.float32)
    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights, q_scale]


@torch.library.custom_op("trtllm::k31_nvfp4_mla_dsa_attn_inplace", mutates_args=("output",))
def mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA.

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale] when
    the indexer ran in Op 1 ([] for a shared-indexer layer). Runs
    sparse_attn_indexer then dispatches context/generation attention. This
    op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = _extract_extra_attrs(layer_idx)
    _forward_dsa_attn(
        mla_layer,
        q,
        compressed_kv,
        k_pe,
        latent_cache,
        indexer_intermediates,
        position_ids,
        metadata,
        output,
    )
    if mla_layer.zero_padding_output_rows and output.shape[0] > metadata.num_tokens:
        output[metadata.num_tokens :].zero_()


@mla_dsa_attn_inplace.register_fake
def _mla_dsa_attn_inplace_fake(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Model the in-place output mutation during fake-tensor propagation."""


maybe_bcg_mla_dsa_attn_inplace = eager_on_graph(mla_dsa_attn_inplace)


# ====================================================================================================================
# DSA hooks of the MLA module
# ====================================================================================================================

# The hooks take the target's MLA module (``mla`` below) without importing the target.


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
    if getattr(self.mqa, "has_fp4_kv_cache", False):
        if latent_cache is None:
            raise ValueError("NVFP4 DSA context requires latent_cache")
        # Append every uncached context token before sparse prediction. The
        # predictor then dequantizes the union of selected TopK rows into a
        # compact FP8 pool, while the attention op skips its normal KV append.
        self.mqa.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata, is_generation=False
        )
        from ...modeling import _mla_forward_absorption_context

        return _mla_forward_absorption_context(
            self,
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            position_ids=position_ids,
            latent_cache=None,
            q_rope_applied=True,
            sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
        )
    from ...modeling import _mla_forward_absorption_context

    return _mla_forward_absorption_context(
        self,
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
    from ...modeling import _mla_forward_absorption_generation

    return _mla_forward_absorption_generation(
        self,
        q,
        compressed_kv,
        k_pe,
        attn_metadata,
        output,
        position_ids=position_ids,
        latent_cache=latent_cache,
        sparse_backend_args=DSABackendForwardArgs(indexer_intermediates=indexer_intermediates),
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

    Returns [q, compressed_kv, k_pe, latent_cache] for a shared-indexer
    layer, else [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8,
    k_scale, weights, q_scale].  q_scale is unused on the FP8 path but
    always present so CUDA graph capture sees a stable 9-tensor shape.
    """
    from ...modeling import _mla_project

    return _mla_project(self, hidden_states, position_ids)


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
    when the indexer ran in Op 1, or [] for a shared-indexer layer.

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

    if self.mqa.indexer is not None:
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
