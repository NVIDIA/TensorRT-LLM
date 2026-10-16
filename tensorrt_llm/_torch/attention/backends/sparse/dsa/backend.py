# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense Sparse Attention (DSA) backend for TRT-LLM with indexer-based TopK selection."""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention.kv_cache_conversion import encode_high_precision_kv
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    MixedPrecisionKVCacheManager,
    Role,
)
from tensorrt_llm.models.modeling_utils import QuantConfig

from .indexer import (
    Indexer,
    transform_local_topk_and_prepare_pool_view,
    transform_local_topk_and_prepare_pool_view_grouped,
)
from .metadata import DSAtrtllmAttentionMetadata
from .params import DSAParams

ModelConfig = tensorrt_llm.bindings.ModelConfig

# Cross-layer fan-out of the DSA index remap (convert_req_index_to_global). Each
# full+shared indexer group's per-layer remap launches collapse into a single
# grouped launch per group (grid.z = group_size), with shared layers consuming a
# precomputed slice; grouped output is bit-identical to the per-layer path.
# Enabled by default; set TRTLLM_DISABLE_DSA_GROUP_REMAP=1 to force the per-layer
# path. Applies only to the generation forward (MLA path); context and MTP draft
# passes keep the per-layer path. Auto-inert on models without shared indexer
# layers (every group is a singleton -> per-layer fallback).
_GROUP_REMAP = os.environ.get("TRTLLM_DISABLE_DSA_GROUP_REMAP", "0") != "1"

_NVFP4_MAX_BLOCK_SCALE = 448.0
_NVFP4_MAX_VALUE = 6.0
_NVFP4_MLA_KV_CACHE_AMAX_ENV = "TRTLLM_NVFP4_MLA_KV_CACHE_AMAX"
_NVFP4_MLA_KV_CACHE_DEFAULT_AMAX = 100.0


def _get_nvfp4_mla_kv_cache_amax() -> float:
    value = os.environ.get(
        _NVFP4_MLA_KV_CACHE_AMAX_ENV,
        str(_NVFP4_MLA_KV_CACHE_DEFAULT_AMAX),
    )
    try:
        amax = float(value)
    except ValueError as error:
        raise ValueError(
            f"{_NVFP4_MLA_KV_CACHE_AMAX_ENV} must be a positive finite float, got {value!r}"
        ) from error
    if not math.isfinite(amax) or amax <= 0:
        raise ValueError(
            f"{_NVFP4_MLA_KV_CACHE_AMAX_ENV} must be a positive finite float, got {value!r}"
        )
    return amax


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
        kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
        self.use_fp8_ds_mla = kv_cache_dtype == "fp8_ds_mla"
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

        # The inherited kv_scale_* tensors are also consumed by the FP8 Q
        # quantization path in TRTLLMGen. Keep those at their default value and
        # use separate scales for the E2M1 x E4M3 KV representation: its global
        # scale contains an extra factor of six that would otherwise saturate Q.
        self._nvfp4_mla_kv_scale_quant_orig = self.kv_scale_quant_orig
        self._nvfp4_mla_kv_scale_orig_quant = self.kv_scale_orig_quant
        if quant_config is not None and quant_config.layer_quant_mode.has_fp4_kv_cache():
            kv_cache_amax = _get_nvfp4_mla_kv_cache_amax()
            dequant_scale = kv_cache_amax / (_NVFP4_MAX_BLOCK_SCALE * _NVFP4_MAX_VALUE)
            self._nvfp4_mla_kv_scale_quant_orig = torch.full_like(
                self.kv_cache_scaling_factor, dequant_scale
            )
            self._nvfp4_mla_kv_scale_orig_quant = torch.full_like(
                self.kv_cache_scaling_factor, 1.0 / dequant_scale
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

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(metadata.kv_cache_manager, MixedPrecisionKVCacheManager):
            # Sparse MLA already gathers the selected rows into its native
            # Attention input; it does not need the dense K/V paged view.
            return self._forward_impl(
                q, k, v, metadata, merge_attention_forward_args(forward_args, kwargs)
            )
        return super().forward(q, k, v, metadata, forward_args, **kwargs)

    def _high_precision_rows(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        positions: torch.Tensor,
        request_indices: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        manager = metadata.kv_cache_manager
        if not isinstance(manager, MixedPrecisionKVCacheManager):
            return None
        if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
            raise NotImplementedError("Mixed-precision DSA currently requires eager execution")
        if not self.has_fp4_kv_cache:
            raise ValueError("Mixed-precision DSA requires the native NVFP4 KV writer")
        pool, tables = manager.get_high_precision_rows(
            self.get_local_layer_idx(metadata), Role.KEY, metadata.request_ids
        )
        width = max(map(len, tables))
        table = torch.tensor(
            [row + [-1] * (width - len(row)) for row in tables],
            dtype=torch.int64,
            device=positions.device,
        )
        blocks = positions.long() // manager.tokens_per_block
        pages = table[request_indices.long(), blocks.clamp(0, width - 1)]
        valid = (positions >= 0) & (blocks < width) & (pages >= 0)
        return pool, pages, positions.long() % manager.tokens_per_block, valid

    def _store_high_precision_kv(
        self,
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        *,
        is_generation: bool,
        apply_rope: bool = False,
    ) -> None:
        if not isinstance(metadata.kv_cache_manager, MixedPrecisionKVCacheManager):
            return
        start = metadata.num_ctx_tokens if is_generation else 0
        end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
        request_indices = metadata.req_idx_per_token[start:end].long()
        lengths = metadata.seq_lens_cuda[: metadata.num_seqs].long()
        starts = lengths.cumsum(0) - lengths
        positions = (
            metadata.kv_lens_cuda[request_indices]
            - lengths[request_indices]
            + torch.arange(start, end, device=latent_cache.device)
            - starts[request_indices]
        )
        pool, pages, offsets, valid = self._high_precision_rows(
            metadata, positions, request_indices
        )
        if apply_rope:
            # The generation writer leaves its input unrotated. Use the same
            # unfused RoPE op as MLA, then store values before NVFP4 conversion.
            from tensorrt_llm._torch.custom_ops import (
                flashinfer_apply_rope_with_cos_sin_cache_inplace,
            )

            latent_cache = latent_cache.clone()
            rope = latent_cache[:, self.mla_params.kv_lora_rank :].contiguous()
            cos_sin = self.rope_params.create_rope_const_params(interleave=False)[1]
            flashinfer_apply_rope_with_cos_sin_cache_inplace(
                positions,
                torch.empty_like(rope),
                rope,
                self.mla_params.qk_rope_head_dim,
                cos_sin.view(self.rope_params.max_positions, -1),
                False,
            )
            latent_cache[:, self.mla_params.kv_lora_rank :] = rope
        values = encode_high_precision_kv(latent_cache[valid], pool.dtype).unsqueeze(1)
        pool.view(torch.uint8)[pages[valid], offsets[valid]] = values.view(torch.uint8)

    def mla_rope_generation(
        self,
        fused_q: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        latent_cache: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        *args,
        **kwargs,
    ) -> None:
        super().mla_rope_generation(fused_q, q_pe, latent_cache, metadata, *args, **kwargs)
        self._store_high_precision_kv(latent_cache, metadata, is_generation=True, apply_rope=True)

    def _overlay_high_precision_kv(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        topk_indices: torch.Tensor,
        compact_indices: torch.Tensor,
        scratch: torch.Tensor,
        *,
        is_generation: bool,
    ) -> None:
        start = metadata.num_ctx_tokens if is_generation else 0
        end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
        rows = self._high_precision_rows(
            metadata, topk_indices, metadata.req_idx_per_token[start:end, None]
        )
        if rows is None:
            return
        pool, pages, offsets, valid = rows
        valid &= compact_indices >= 0
        source = valid.flatten().nonzero().flatten()
        destinations, inverse = torch.unique(
            compact_indices.flatten()[source].long(), return_inverse=True
        )
        # Context queries can select the same compact row many times. Read
        # and convert each high-precision row only once.
        selected = torch.empty_like(destinations).scatter_(0, inverse, source)
        values = pool.view(torch.uint8)[pages.flatten()[selected], offsets.flatten()[selected]]
        # Native sparse NVFP4 Attention consumes FP8, including high-pool rows.
        values = encode_high_precision_kv(values.view(pool.dtype), scratch.dtype)
        scratch.view(-1, 1, self.head_dim).view(torch.uint8)[destinations] = values.view(
            torch.uint8
        )

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
        kv_cache_dtype = getattr(metadata.kv_cache_manager, "dtype", None)
        if kv_cache_dtype == tensorrt_llm.bindings.DataType.NVFP4 and not is_generation:
            metadata._ensure_pool_view_cached()
            page_index_scale, layer_offset = (
                metadata.kv_cache_manager.get_primary_pool_page_index_params(local_layer_idx)
            )
            total_kv_tokens = metadata.num_ctx_mla_kv_tokens
            if total_kv_tokens <= 0:
                raise RuntimeError("NVFP4 DSA context has no active KV tokens")
            head_dim = metadata.kv_cache_manager.head_dim
            # Static sparse FMHA may prefetch up to TopK rows even when a
            # short context has fewer valid KV tokens. The compaction itself
            # can select at most min(total KV tokens, query/TopK pairs), so a
            # late chunk with a long cached prefix does not need one fp8 row
            # for every token in that prefix.
            max_selected_tokens = min(total_kv_tokens, topk_indices.numel())
            scratch_capacity = max(max_selected_tokens, topk_indices.shape[1])
            # The previous layer's attention has already been enqueued on this
            # stream. Drop its Python reference before allocating the next
            # layer's scratch so the caching allocator can reuse that storage
            # in stream order instead of transiently keeping two full buffers.
            metadata.nvfp4_mla_context_fp8_scratch = None
            scratch = torch.empty(
                (scratch_capacity, 1, head_dim),
                dtype=torch.float8_e4m3fn,
                device=topk_indices.device,
            )
            compact_indices = torch.empty_like(topk_indices)
            torch.ops.trtllm.nvfp4_mla_context_kv_cache_gather(
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                topk_indices,
                metadata._cached_req_idx_ctx,
                metadata._cached_block_table_ctx,
                metadata.ctx_kv_indptr[: metadata.num_contexts + 1],
                scratch,
                compact_indices,
                self._nvfp4_mla_kv_scale_quant_orig,
                local_layer_idx,
                total_kv_tokens,
                metadata._cached_tokens_per_block,
                page_index_scale * metadata._cached_tokens_per_block,
                layer_offset,
                metadata.kv_cache_manager.mla_kv_cache_residual_dim,
                metadata._cached_num_pool_tokens,
            )
            self._overlay_high_precision_kv(
                metadata, topk_indices, compact_indices, scratch, is_generation=False
            )
            metadata.nvfp4_mla_context_fp8_scratch = scratch
            forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr = scratch.data_ptr()
            return compact_indices, None

        topk_indices_global = self._remap_topk_to_global(topk_indices, metadata, is_generation)

        if kv_cache_dtype == tensorrt_llm.bindings.DataType.NVFP4:
            scratch = metadata.nvfp4_mla_fp8_scratch
            compact_indices = metadata.nvfp4_mla_compact_indices
            if scratch is None or compact_indices is None:
                raise RuntimeError("NVFP4 MLA scratch buffers were not allocated")
            num_rows = topk_indices_global.shape[0]
            if num_rows > scratch.shape[0] or num_rows > compact_indices.shape[0]:
                raise RuntimeError(
                    "NVFP4 MLA generation rows exceed the allocated MTP "
                    f"workspace: requested {num_rows}, capacity {scratch.shape[0]}"
                )
            scratch = scratch[:num_rows]
            compact_indices = compact_indices[:num_rows]
            torch.ops.trtllm.nvfp4_mla_kv_cache_gather(
                metadata.host_kv_cache_pool_pointers,
                metadata.host_kv_cache_pool_mapping,
                topk_indices_global,
                scratch,
                compact_indices,
                self._nvfp4_mla_kv_scale_quant_orig,
                local_layer_idx,
                metadata.kv_cache_manager.mla_kv_cache_residual_dim,
                metadata._cached_num_pool_tokens,
            )
            self._overlay_high_precision_kv(
                metadata, topk_indices, compact_indices, scratch, is_generation=True
            )
            # Static sparse MLA treats indices as offsets from kvPtr. Feed it
            # compact offsets and the FP8 scratch base; TRTLLMGen is unchanged.
            forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr = scratch.data_ptr()
            return compact_indices, None

        return topk_indices_global, None

    def _remap_topk_to_global(
        self,
        topk_indices: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        is_generation: bool,
    ) -> torch.Tensor:
        """Convert this layer's local top-k to global pool indices.

        Uses the cross-layer fan-out (grouped) remap when enabled and applicable;
        otherwise the per-layer single-op path (unchanged).
        """
        local_layer_idx = self.get_local_layer_idx(metadata)

        # Grouping applies only to the MLA target-verify generation forward.
        # Context and MTP draft passes keep the per-layer path (draft passes may
        # reuse a frozen top-k for shared layers that differs from the leader's,
        # so fan-out is unsafe there; context is excluded to avoid a group-sized
        # allocation over many tokens).
        if _GROUP_REMAP and is_generation and not metadata.in_mtp_draft_loop:
            grouped = self._grouped_remap_topk_to_global(
                topk_indices, metadata, local_layer_idx, is_generation
            )
            if grouped is not None:
                return grouped

        topk_indices_global, _ = transform_local_topk_and_prepare_pool_view(
            topk_indices, metadata, local_layer_idx, is_generation
        )
        return topk_indices_global

    def _grouped_remap_topk_to_global(
        self,
        topk_indices: torch.Tensor,
        metadata: DSAtrtllmAttentionMetadata,
        local_layer_idx: int,
        is_generation: bool,
    ) -> Optional[torch.Tensor]:
        """Grouped fan-out remap for one full+shared indexer group.

        The full (leader) layer computes the batched remap for the whole group in
        one launch and caches it; each shared layer consumes its precomputed
        slice. Returns None to signal a fall back to the per-layer path (inactive
        group, or leader output not yet available).

        Grouped output is bit-identical to the per-layer path by construction;
        that equivalence is covered by unit tests (see
        ``tests/unittest/_torch/attention/sparse/dsa/test_cpp_custom_ops.py``).
        """
        struct = metadata._ensure_group_remap_struct()
        leader_of = struct.get("leader_of")
        if leader_of is None or local_layer_idx >= len(leader_of):
            return None
        leader = leader_of[local_layer_idx]
        if leader < 0 or not struct["group_active"].get(leader, False):
            return None
        slot = struct["slot_of"][local_layer_idx]

        if self.indexer is not None:
            # Leader (full-indexer layer): compute the whole group in one launch.
            batched = transform_local_topk_and_prepare_pool_view_grouped(
                topk_indices,
                metadata,
                struct["group_layer_ids"][leader],
                struct["group_scale"][leader],
                is_generation,
            )
            metadata._group_remap_batched[leader] = batched
        else:
            # Shared layer: read the leader's precomputed batch. The batch is
            # cleared at every step boundary, so a present entry is from this
            # forward; still require it to match this layer's group slot and
            # top-k shape, and otherwise fall back to the per-layer path. This
            # makes any leader/follower ordering or shape mismatch a safe
            # per-layer fallback rather than a stale/out-of-bounds read.
            batched = metadata._group_remap_batched.get(leader, None)
            if (
                batched is None
                or slot >= batched.shape[0]
                or batched.shape[1:] != topk_indices.shape
            ):
                return None

        return batched[slot]

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
        local_layer_idx = self.get_local_layer_idx(metadata)
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
            self._nvfp4_mla_kv_scale_orig_quant
            if metadata.kv_cache_manager.dtype == tensorrt_llm.bindings.DataType.NVFP4
            else None,
            metadata.kv_cache_manager.mla_kv_cache_residual_dim,
            local_layer_idx,
            metadata.kv_cache_manager.tokens_per_block,
            metadata.kv_cache_manager.max_seq_len,
            beam_width,
            self.quant_mode,
        )
        self._store_high_precision_kv(latent_cache, metadata, is_generation=is_generation)
