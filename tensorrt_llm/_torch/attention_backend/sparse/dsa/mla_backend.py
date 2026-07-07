# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA-specific hooks for the shared MLA module."""

from typing import List, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.modules.multi_stream_utils import maybe_execute_in_parallel
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm._utils import get_sm_version, nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger

from .indexer import transform_local_topk_and_prepare_pool_view
from .metadata import DSAtrtllmAttentionMetadata

try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None


def _fp8_block_scaling_bmm_out(*args, **kwargs):
    from tensorrt_llm._torch.modules.mla import fp8_block_scaling_bmm_out

    return fp8_block_scaling_bmm_out(*args, **kwargs)


def _forward_context_sparse_mla(*args, **kwargs):
    from ..mla import forward_context_sparse_mla

    return forward_context_sparse_mla(*args, **kwargs)


def _forward_generation_sparse_mla(*args, **kwargs):
    from ..mla import forward_generation_sparse_mla

    return forward_generation_sparse_mla(*args, **kwargs)


def forward_impl_with_dsa(
    self,
    position_ids: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
) -> None:
    """
    Forward pass for the MLA module with DSA (always in MQA mode).
    Writes result into output tensor in-place.

    Delegates to forward_dsa_proj (token-wise projections) followed by
    forward_dsa_attn (batch-dependent attention dispatch).

    Args:
        position_ids (Optional[torch.IntTensor]): The position IDs.
        hidden_states (torch.Tensor): The hidden states.
        attn_metadata (AttentionMetadata): The attention metadata.
        output (torch.Tensor): The output tensor to write results into.
    """
    proj_outputs = forward_dsa_proj(self, position_ids, hidden_states, attn_metadata)
    q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
    indexer_intermediates = proj_outputs[4:]
    forward_dsa_attn(
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
    to num_tokens happens in forward_dsa_attn (Op 2, outside graph).

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

    # DSA "shared" layer: no indexer; reuses the previous full layer's
    # top-k (in forward_dsa_attn), so skip the projection.
    if self.mqa.indexer is None:
        return [q, compressed_kv, k_pe, latent_cache]

    # pre_indexer_proj is the CUDA-graph-safe portion: pure token-wise
    # compute (cublas_mm, rope, FP4/FP8 quantize, weight scaling) with no
    # access to batch-specific metadata or the k cache. Returns q_scale
    # as a 5th element so the FP4 dispatch can forward it to the kernel;
    # the FP8 path ignores it in forward_dsa_attn.
    q_fp8, k_fp8, k_scale, weights, q_scale = self.mqa.indexer.pre_indexer_proj(
        qr, hidden_states, position_ids
    )

    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights, q_scale]


def forward_dsa_attn(
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

    if use_short_mha_for_ctx and num_generations == 0:
        topk_indices = None
    elif self.mqa.indexer is None:
        # DSA "shared" layer: reuse the previous full layer's top-k. These
        # are local token positions, so they are layer-agnostic; each layer
        # applies its own paged-KV transform downstream.
        topk_indices = getattr(attn_metadata, "shared_topk_indices", None)
        assert topk_indices is not None, (
            "DSA shared layer has no top-k from a preceding full indexer "
            "layer; check the index_topk_pattern/freq schedule."
        )
    else:
        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        # Slice indexer intermediates to actual num_tokens (they were
        # computed on the full padded tensor in Op 1).
        q_fp8 = q_fp8[:num_tokens, ...]
        k_fp8 = k_fp8[:num_tokens, ...]
        k_scale = k_scale[:num_tokens, ...]
        weights = weights[:num_tokens, ...]
        q_scale = q_scale[:num_tokens, ...]
        topk_indices = self.mqa.indexer.sparse_attn_indexer(
            attn_metadata,
            q,  # only used for shape/device in buffer allocation
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            q_scale=q_scale,
        )
        # Stash for subsequent DSA "shared" layers (full -> shared reuse);
        # unused by a dense per-layer indexer.
        attn_metadata.shared_topk_indices = topk_indices

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

        _forward_context_sparse_mla(
            self,
            q_ctx,
            compressed_kv_ctx,
            k_pe_ctx,
            attn_metadata,
            output[:num_ctx_tokens, :],
            latent_cache_ctx,
            topk_indices=topk_indices[:num_ctx_tokens, :] if topk_indices is not None else None,
            position_ids=ctx_position_ids,
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

        _forward_generation_sparse_mla(
            self,
            q_gen,
            compressed_kv_gen,
            k_pe_gen,
            attn_metadata,
            output[num_ctx_tokens:num_tokens, :],
            latent_cache=latent_cache_gen,
            topk_indices=topk_indices[num_ctx_tokens:num_tokens, :],
            position_ids=gen_position_ids,
        )


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
        and not self.apply_rotary_emb
        and self.mapping.cp_size == 1
        and position_ids is not None
    ):
        return False
    effective_len = getattr(attn_metadata, "max_ctx_kv_len", attn_metadata.num_ctx_tokens)
    return effective_len <= self.short_seq_mha_threshold


@nvtx_range("forward_sparse_mla_kvcache_bf16")
def forward_sparse_mla_kvcache_bf16(
    self,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    output: torch.Tensor,
    topk_indices: torch.Tensor,
    is_generation: bool = False,
) -> torch.Tensor:
    """
    Forward sparse MLA (DSA) for BF16 KV cache for both context and generation phases using FlashMLA kernels

    To form the input for FlashMLA kernel and adapt our KV cache manager, we need to:
    1. Append current tokens to paged cache and apply rope to q/k via mla_rope_append_paged_kv_assign_q
    2. Load full kv cache from paged memory (with k rope applied)
    3. Call FlashMLA sparse attention kernel for sparse prefill/decode
    """
    assert isinstance(attn_metadata, DSAtrtllmAttentionMetadata), (
        "DSA requires DSAtrtllmAttentionMetadata"
    )
    # Append current tokens to paged cache and apply RoPE to q
    # This writes latent_cache to paged KV and modifies q in-place
    trtllm_attention = self.mqa
    with nvtx_range_debug(f"mla_rope_append_paged_kv_assign_q_is_generation={is_generation}"):
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata, is_generation=is_generation
        )

    num_tokens = q.shape[0]
    q_nope, q_rope = q.view(-1, self.num_heads_tp, self.qk_head_dim).split(
        [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
    )
    q_nope_out = torch.empty(
        [num_tokens, self.num_heads_tp, (self.kv_lora_rank)],
        dtype=q.dtype,
        device=q.device,
    )

    if self.k_b_proj_trans.dtype == torch.bfloat16:
        # [num_heads, num_tokens, self.qk_nope_head_dim]
        q_nope_t = q_nope.transpose(0, 1)
        # [num_heads, num_tokens, self.kv_lora_rank]
        q_nope_out = q_nope_out.transpose(0, 1)

        # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
        # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
        # The output of bmm is written directly into fused_q
        self._bmm_bf16_out(
            q_nope_t, self.k_b_proj_trans, self.k_b_proj_trans.transpose(1, 2), q_nope_out
        )
    elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
        # [num_heads, num_tokens, self.kv_lora_rank]
        q_nope_out = q_nope_out.transpose(0, 1)

        _fp8_block_scaling_bmm_out(
            q_nope,
            self.k_b_proj_trans,
            self.k_b_proj_trans_scale,
            q_nope_out,
            self.k_b_proj_trans_dequant,
            self.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

    q_nope_out = q_nope_out.transpose(0, 1)
    q_concat = torch.cat([q_nope_out, q_rope], dim=-1)

    sm_version = get_sm_version()
    # FlashMLA sparse kernel (bf16) requires num_heads=128 on sm100 or multiple of 64 on sm90
    if sm_version >= 100:
        padding = 128
        assert self.num_heads_tp <= padding, (
            f"SM100 FlashMLA sparse kernel requires exactly {padding} heads, "
            f"got {self.num_heads_tp}. Padding from values > {padding} is not supported."
        )
    else:  # SM90
        padding = ((self.num_heads_tp + 63) // 64) * 64  # multiple of 64

    if self.num_heads_tp != padding:
        logger.warning_once(
            f"Padding num_heads from {self.num_heads_tp} to {padding} "
            f"due to FlashMLA sparse attention kernel requirement",
            key="sparse_mla_padding_warning",
        )

        # Create padded tensor with zeros for extra heads
        q_padded = q_concat.new_empty((num_tokens, padding, q_concat.shape[2]))
        q_padded[:, : self.num_heads_tp, :] = q_concat
        q_concat = q_padded

    # Convert indices and return all-layer KV pool
    # The pool is layer-interleaved. Return the all-layer view and adjust
    # top-k indices by num_layers * tokens_per_block to avoid a per-layer copy.
    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        attn_metadata,
        layer_idx=self.layer_idx,
        is_generation=is_generation,
    )
    topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
    if flash_mla_sparse_fwd is not None:
        attn_out_latent = flash_mla_sparse_fwd(
            q_concat, kv_cache_pool, topk_indices_pool, self.softmax_scale
        )[0]
    else:
        raise RuntimeError(
            "flash_mla_sparse_fwd not available. Please ensure FlashMLA module is built."
        )

    # [seq, num_heads, kv_lora_rank], account for padding
    attn_out_latent = attn_out_latent[:, : self.num_heads_tp, :]
    attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp, self.kv_lora_rank])
    if self.num_heads_tp != padding:
        attn_out_latent = attn_out_latent.contiguous()

    assert attn_out_latent.shape[0] == q.shape[0] and attn_out_latent.shape[1] == self.num_heads_tp

    attn_output = output.view([num_tokens, self.num_heads_tp, self.v_head_dim])

    if self.v_b_proj.dtype == torch.bfloat16:
        # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
        # -> [num_heads, seq, v_head_dim]
        self._bmm_bf16_out(
            attn_out_latent.transpose(0, 1),
            self.v_b_proj,
            self.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
        )
    elif self.v_b_proj.dtype == torch.float8_e4m3fn:
        _fp8_block_scaling_bmm_out(
            attn_out_latent,
            self.v_b_proj,
            self.v_b_proj_scale,
            attn_output.transpose(0, 1),
            self.v_b_proj_dequant,
            self.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")
    return output
