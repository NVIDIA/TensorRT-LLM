# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer sparse-MLA attention backend for DSA models on SM120/SM121.

On SM120 the C++ attention op compiles out the trtllm-gen sparse-MLA kernels
for both phases (``attentionOp.h`` excludes SM120 from ``mUseTllmGen``), so
this backend routes DSA (DeepSeek-V3.2 / GLM ``glm_moe_dsa``) attention
through flashinfer's SM120 sparse MLA kernels instead. It reuses the whole
DSA prediction stack from the TRTLLM backend — metadata, indexer (including
cross-layer sharing), and the local→global index conversion — and replaces
only the attention call and the KV append:

* the main latent pool holds inline-scale pages (``inline_scale_kv``, the
  FlashMLA ABI flashinfer's V32-family kernels read; GLM-style arbitrary
  FP32 scales);
* RoPE runs in Python (``support_fused_rope() -> False``), and the roped
  latent rows are quantized and scattered into the pool here rather than
  inside the C++ ``mla_rope_append_paged_kv_assign_q`` op;
* one flashinfer call serves context and generation alike — the kernel
  auto-dispatches its decode fast path (<= 64 query tokens) vs the prefill
  orchestrator, and ``-1`` index padding carries the per-token top-k
  semantics the indexer already emits.
"""

import math
from typing import Optional

import torch

from ..interface import (AttentionForwardArgs, AttentionInputType,
                         merge_attention_forward_args)
from . import inline_scale_kv
from .dsa import (DSAtrtllmAttentionMetadata, DSATrtllmAttention,
                  transform_local_topk_and_prepare_pool_view)

_KV_SPLIT_TILE = 64  # BLOCK_SIZE_N of the SM120 kernels; sizes split-K scratch


def _sparse_mla_op():
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    return _sparse_mla_sm120_paged_attention


def _inline_scale_pool_paged(metadata: DSAtrtllmAttentionMetadata) -> torch.Tensor:
    """Whole-pool uint8 view [num_pages, page_size, 656] anchored at slot 0.

    Global slot ids from ``convert_req_index_to_global`` address pages of the
    layer-interleaved primary pool (page ordinal = block * num_layers +
    layer), so one flat view serves every layer. Keyed by the manager's
    identity: a metadata object can outlive a manager swap (KV-cache
    estimation builds a throwaway manager) and must not serve stale views.
    """
    manager = metadata.kv_cache_manager
    cached = getattr(metadata, "_inline_scale_pool", None)
    if cached is not None and cached[0] == id(manager):
        return cached[1]
    pool = manager.get_unique_primary_pool()
    num_blocks, num_layers = pool.shape[0], pool.shape[1]
    page_bytes = manager.tokens_per_block * inline_scale_kv.TOKEN_BYTES
    view = pool.view(torch.uint8).reshape(num_blocks * num_layers, page_bytes)
    assert view.is_contiguous()
    paged = view.view(-1, manager.tokens_per_block, inline_scale_kv.TOKEN_BYTES)
    metadata._inline_scale_pool = (id(manager), paged)
    return paged


class DSAFlashInferAttention(DSATrtllmAttention):
    """DSA sparse MLA served by flashinfer's SM120 inline-scale kernels."""

    Metadata = DSAtrtllmAttentionMetadata
    kv_token_layout = "inline_scale"

    def support_fused_rope(self) -> bool:
        # RoPE runs in mla.py; this backend receives roped q and a latent
        # cache whose rope half the model layer has already resynced. This
        # also disables the short-seq MHA context bypass (it requires fused
        # rope), keeping every sequence on the sparse path this backend
        # implements.
        return False

    def _latent_append(
        self,
        metadata: DSAtrtllmAttentionMetadata,
        latent_rows: torch.Tensor,
        start_idx: int,
        end_idx: int,
        is_generation: bool,
    ) -> None:
        """Quantize the new latent rows and scatter them into the main pool.

        Write slots come from the same ``convert_req_index_to_global`` op the
        top-k conversion uses, fed each token's own position — write and read
        sides share one slot currency by construction.
        """
        metadata._ensure_pool_view_cached()
        positions = metadata.token_positions_cuda[start_idx:end_idx]
        if is_generation:
            block_table = metadata._cached_block_table_gen
            req_idx = metadata._cached_req_idx_gen
        else:
            block_table = metadata._cached_block_table_ctx
            req_idx = metadata._cached_req_idx_ctx
        loc = torch.ops.trtllm.convert_req_index_to_global(
            req_idx,
            block_table,
            positions.unsqueeze(1).contiguous(),
            metadata._cached_tokens_per_block,
            1,
            metadata._cached_stride_factor,
            self.get_local_layer_idx(metadata),
        ).view(-1)
        pool = _inline_scale_pool_paged(metadata)
        inline_scale_kv.quant_scatter(
            pool.view(pool.shape[0], -1),
            loc,
            latent_rows,
            page_size=metadata._cached_tokens_per_block,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DSAtrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ):
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if metadata.max_draft_tokens > 0:
            raise NotImplementedError(
                "MTP / speculative decoding is not supported by the FlashInfer "
                "DSA backend yet.")

        attention_input_type = forward_args.attention_input_type
        if attention_input_type == AttentionInputType.context_only:
            start_idx, end_idx = 0, metadata.num_ctx_tokens
            is_generation = False
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx, end_idx = metadata.num_ctx_tokens, metadata.num_tokens
            is_generation = True
        else:
            raise NotImplementedError(
                "The FlashInfer DSA backend expects phase-split calls "
                "(context_only / generation_only), matching the DSA model "
                "forward structure.")
        num_tokens = end_idx - start_idx

        # The absorption path passes output=None for non-DSv4 DSA models and
        # consumes the returned latent as [num_tokens, heads * kv_lora_rank]
        # (mla.py's post-attention assert + v_b_proj bmm contract).
        kv_lora_rank = self.mla_params.kv_lora_rank
        output = forward_args.output
        if output is None:
            output = q.new_empty(num_tokens, self.num_heads * kv_lora_rank)
        else:
            assert output.numel() == num_tokens * self.num_heads * kv_lora_rank, (
                "FlashInfer DSA backend got a preallocated output of "
                f"{output.numel()} elements; expected "
                f"{num_tokens} x {self.num_heads} x {kv_lora_rank}")
        if num_tokens == 0:
            return output

        q_view = q.view(num_tokens, self.num_heads, self.head_dim)
        if q_view.dtype != torch.bfloat16:
            raise NotImplementedError(
                "FlashInfer SM120 sparse MLA takes bf16 queries; fused FP8-Q "
                "paths must stay disabled for this backend.")

        # Append the new latent rows before attention: each token attends to
        # itself, and the indexer's top-k covers the current position.
        latent_cache = forward_args.latent_cache
        if latent_cache is not None and forward_args.update_kv_cache:
            self._latent_append(
                metadata,
                latent_cache.view(num_tokens, self.head_dim),
                start_idx,
                end_idx,
                is_generation,
            )

        if forward_args.topk_indices is None:
            raise NotImplementedError(
                "The FlashInfer DSA backend serves sparse attention only; a "
                "dense (topk_indices=None) call implies a bypass path that "
                "must stay disabled with this backend.")
        topk_indices_global, _ = self.sparse_attn_predict(
            q, k, metadata, forward_args)
        if not topk_indices_global.is_contiguous():
            topk_indices_global = topk_indices_global.contiguous()
        topk = topk_indices_global.shape[-1]

        out_view = output.view(num_tokens, self.num_heads, kv_lora_rank)
        out_lse = torch.zeros(num_tokens,
                              self.num_heads,
                              dtype=torch.float32,
                              device=q.device)
        if num_tokens <= 64:
            # Split-K scratch must be initialized: per-token top-k truncation
            # can leave splits the kernel never writes, and the merge reads
            # every split slot. A -inf LSE makes an unwritten split weightless.
            num_splits = (topk + _KV_SPLIT_TILE - 1) // _KV_SPLIT_TILE
            mid_out = torch.zeros(
                num_tokens,
                self.num_heads,
                num_splits,
                self.mla_params.kv_lora_rank,
                dtype=torch.bfloat16,
                device=q.device,
            )
            mid_lse = torch.full(
                (num_tokens, self.num_heads, num_splits),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            mid_out = None
            mid_lse = None

        # The softmax scale uses the per-head qk width, not the absorbed
        # latent width in self.head_dim — they coincide for DSv4 (512) but
        # differ here (GLM: 256 vs 576).
        qk_head_dim = (self.mla_params.qk_nope_head_dim +
                       self.mla_params.qk_rope_head_dim)
        sm_scale = 1.0 / (self.q_scaling * math.sqrt(qk_head_dim))

        _sparse_mla_op()(
            q_view.contiguous(),
            _inline_scale_pool_paged(metadata),
            topk_indices_global,
            out_view,
            out_lse,
            sm_scale,
            d_v=self.mla_params.kv_lora_rank,
            kv_scale_format="arbitrary_fp32",
            mid_out=mid_out,
            mid_lse=mid_lse,
        )
        return output
