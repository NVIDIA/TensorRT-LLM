# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm.models.modeling_utils import QuantConfig

from .indexer import Indexer, transform_local_topk_and_prepare_pool_view
from .metadata import DSAtrtllmAttentionMetadata
from .params import DSAParams

ModelConfig = tensorrt_llm.bindings.ModelConfig


class DSATrtllmAttention(TrtllmAttention):
    """TRT-LLM attention layer with DSA sparse indexer for MLA models."""

    Metadata = DSAtrtllmAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        mla_params: Optional[MLAParams] = None,
        skip_create_weights_in_init: bool = False,
        attention_chunk_size: Optional[int] = None,
        sparse_params: Optional[DSAParams] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        """Initialize DSA attention with an Indexer sub-module for sparse TopK selection."""
        sparse_attention_config = kwargs.pop("sparse_attention_config", None)
        self.sparse_attention_config = sparse_attention_config
        if (
            sparse_params is None
            and sparse_attention_config is not None
            and hasattr(sparse_attention_config, "to_sparse_params")
        ):
            sparse_params = sparse_attention_config.to_sparse_params(layer_idx=layer_idx)
        if sparse_params is None:
            raise ValueError("sparse_params is required for DSATrtllmAttention and cannot be None")
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            sparse_params=sparse_params,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            q_scaling=q_scaling,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=skip_create_weights_in_init,
            attention_chunk_size=attention_chunk_size,
            **kwargs,
        )

        # Cross-layer indexer sharing: only "full" layers own an indexer;
        # "shared" layers reuse the previous full layer's top-k (see
        # MLA.forward_dsa_*). Resolved per-layer in to_sparse_params; defaults to
        # full (dense per-layer indexer). indexer=None also makes the weight
        # loader skip the (absent) shared-layer indexer weights.
        self.is_full_indexer_layer = getattr(sparse_params, "is_full_indexer_layer", True)
        if self.is_full_indexer_layer:
            self.indexer = Indexer(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_params,
                dtype=dtype,
                layer_idx=layer_idx,
                aux_stream=aux_stream,
            )
        else:
            self.indexer = None

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the DSA indexer and convert its local TopK to paged indices."""
        is_generation = forward_args.attention_input_type == AttentionInputType.generation_only
        sparse_backend_args = forward_args.sparse_backend_args
        phase_start = metadata.num_ctx_tokens if is_generation else 0
        phase_end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
        shared_topk_indices = metadata.shared_topk_indices
        if self.indexer is None:
            topk_indices = shared_topk_indices[phase_start:phase_end]
        else:
            indexer_intermediates = sparse_backend_args.indexer_intermediates
            topk_indices = self.indexer.forward_from_projected(
                metadata,
                q,
                indexer_intermediates,
                is_generation=is_generation,
            )
            preserve_mtp_topk = metadata.in_mtp_draft_loop and self.indexer.mtp_index_share
            if shared_topk_indices is not None and not preserve_mtp_topk:
                shared_topk_indices[
                    phase_start : phase_start + topk_indices.shape[0],
                    : topk_indices.shape[1],
                ].copy_(topk_indices)

        local_layer_idx = self.get_local_layer_idx(metadata)
        if (
            metadata.kv_cache_manager.dtype == tensorrt_llm.bindings.DataType.NVFP4
            and not is_generation
        ):
            metadata._ensure_pool_view_cached()
            page_index_scale, layer_offset = (
                metadata.kv_cache_manager.get_primary_pool_page_index_params(local_layer_idx)
            )
            total_kv_tokens = metadata.num_ctx_mla_kv_tokens
            if total_kv_tokens <= 0:
                raise RuntimeError("NVFP4 DSA context has no active KV tokens")
            head_dim = metadata.kv_cache_manager.head_dim
            # Static sparse FMHA may prefetch up to TopK rows even when a
            # short context has fewer valid KV tokens.
            scratch_capacity = max(total_kv_tokens, topk_indices.shape[1])
            scratch = torch.empty(
                (scratch_capacity, 1, head_dim),
                dtype=torch.float8_e4m3fn,
                device=topk_indices.device,
            )
            gather_scratch = (
                scratch if scratch_capacity == total_kv_tokens else scratch[:total_kv_tokens]
            )
            compact_indices = torch.empty_like(topk_indices)
            torch.ops.trtllm.nvfp4_mla_context_kv_cache_gather(
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                topk_indices,
                metadata._cached_req_idx_ctx,
                metadata._cached_block_table_ctx,
                metadata.ctx_kv_indptr[: metadata.num_contexts + 1],
                gather_scratch,
                compact_indices,
                self.kv_scale_quant_orig,
                local_layer_idx,
                total_kv_tokens,
                metadata._cached_tokens_per_block,
                page_index_scale * metadata._cached_tokens_per_block,
                layer_offset,
                metadata.kv_cache_manager.mla_kv_cache_residual_dim,
                metadata._cached_num_pool_tokens,
            )
            metadata.nvfp4_mla_context_fp8_scratch = scratch
            forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr = scratch.data_ptr()
            return compact_indices, None

        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            topk_indices, metadata, local_layer_idx, is_generation
        )

        if metadata.kv_cache_manager.dtype == tensorrt_llm.bindings.DataType.NVFP4:
            if topk_indices_global.shape[0] != metadata.num_generations:
                raise NotImplementedError(
                    "NVFP4 DSA cache currently supports exactly one query "
                    "token per generation request (MTP is not supported)."
                )
            scratch = metadata.nvfp4_mla_fp8_scratch
            compact_indices = metadata.nvfp4_mla_compact_indices
            if scratch is None or compact_indices is None:
                raise RuntimeError("NVFP4 MLA scratch buffers were not allocated")
            num_rows = topk_indices_global.shape[0]
            scratch = scratch[:num_rows]
            compact_indices = compact_indices[:num_rows]
            torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                topk_indices_global,
                scratch,
                compact_indices,
                self.kv_scale_quant_orig,
                local_layer_idx,
                metadata.kv_cache_manager.mla_kv_cache_residual_dim,
                metadata._cached_num_pool_tokens,
            )
            # Static sparse MLA treats indices as offsets from kvPtr. Feed it
            # compact offsets and the FP8 scratch base; TRTLLMGen is unchanged.
            forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr = scratch.data_ptr()
            return compact_indices, None

        return topk_indices_global, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """No-op KV prediction; DSA uses indexer-based selection instead."""
        return None, None

    def mla_rope_append_paged_kv_assign_q(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool = False,
        **kwargs,
    ) -> None:
        """Apply RoPE, append latent cache to paged KV, and assign query for MLA."""
        if is_generation:
            cached_token_indptr = metadata.gen_cached_token_indptr
            kv_indptr = metadata.gen_kv_indptr
            num_seqs = metadata.num_generations
            max_seq_len = metadata.max_gen_seq_len
            block_offsets = metadata.kv_cache_block_offsets[:, metadata.num_contexts :]
        else:
            cached_token_indptr = metadata.ctx_cached_token_indptr
            kv_indptr = metadata.ctx_kv_indptr
            num_seqs = metadata.num_contexts
            max_seq_len = metadata.max_ctx_seq_len
            block_offsets = metadata.kv_cache_block_offsets
        assert self.is_mla_enable and self.mla_params is not None
        assert metadata.kv_cache_manager is not None

        beam_width = 1

        torch.ops.trtllm.mla_rope_append_paged_kv_assign_q(
            q,
            latent_cache,
            num_seqs,
            cached_token_indptr,
            kv_indptr,
            max_seq_len,
            self.rotary_cos_sin,
            self.num_heads,
            self.mla_params.qk_nope_head_dim,
            self.mla_params.qk_rope_head_dim,
            self.mla_params.kv_lora_rank,
            block_offsets,
            metadata.kv_cache_manager.kv_cache_pool_pointers,
            metadata.kv_cache_manager.kv_cache_pool_mapping,
            self.kv_scale_orig_quant
            if metadata.kv_cache_manager.dtype == tensorrt_llm.bindings.DataType.NVFP4
            else None,
            metadata.kv_cache_manager.mla_kv_cache_residual_dim,
            self.get_local_layer_idx(metadata),
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )
