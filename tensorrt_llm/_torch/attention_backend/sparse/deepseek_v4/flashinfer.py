# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FlashInfer sparse-MLA attention backend for DeepSeek-V4 on SM120/SM121.

On SM120 the C++ attention op compiles out the trtllm-gen sparse-MLA kernels
for both phases (``attentionOp.h`` excludes SM120 from ``mUseTllmGen``), so
this backend routes DeepSeek-V4 attention through flashinfer's packed sparse
MLA kernels instead. It reuses the whole DeepSeek-V4 prediction stack from the
TRTLLM backend — metadata, indexer, compressor, and the local→global index
conversion — and replaces only the attention call and the KV append:

* both pools hold footer-scale pages (``footer_scale_kv``);
* RoPE runs in Python (``support_fused_rope() -> False``), and the roped
  latent rows are quantized and scattered into the SWA pool here rather than
  inside the C++ op;
* one flashinfer call serves context and generation alike — the kernel
  auto-dispatches its decode fast path (<= 64 query tokens) vs the prefill
  orchestrator, and per-token top-k lengths plus ``-1`` index padding carry
  the phase semantics.
"""

import math
from typing import Optional

import torch

from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor
from tensorrt_llm.bindings import DataType

from ...interface import (
    AttentionForwardArgs,
    AttentionInputType,
    merge_attention_forward_args,
)
from ..kernel import deepseek_v4_local_to_global_indices
from .deepseek_v4 import (
    DeepseekV4AttentionType,
    DeepseekV4TrtllmAttention,
    DeepseekV4TrtllmAttentionMetadata,
    get_token_bytes,
)
from . import footer_scale_kv

_KV_SPLIT_TILE = 64  # BLOCK_SIZE_N of the SM120 kernels; sizes split-K scratch


def _sparse_mla_op():
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention

    return _sparse_mla_sm120_paged_attention


def _footer_scale_pool_2d(
    metadata: DeepseekV4TrtllmAttentionMetadata,
    attn_type: DeepseekV4AttentionType,
    compress_ratio: int,
) -> torch.Tensor:
    """Whole-pool uint8 view [num_pages, page_size*584] anchored at slot 0.

    The sparse indices are pool-base-relative (layer offsets are folded into
    the block tables / buffer-pointer arithmetic), so the kernel view must
    span every layer's pages from the pool base. Cached on the metadata
    object, keyed by the cache manager's identity: pool addresses are
    constant for a manager's lifetime, but a metadata object that outlives a
    manager swap (KV-cache estimation builds a throwaway manager) must not
    serve views into the old manager's freed pools.
    """
    manager = metadata.kv_cache_manager
    cache = getattr(metadata, "_footer_scale_pools", None)
    if cache is None:
        cache = {}
        metadata._footer_scale_pools = cache
    key = (id(manager), attn_type, compress_ratio)
    pool = cache.get(key)
    if pool is not None:
        return pool
    if attn_type == DeepseekV4AttentionType.SWA:
        base_ptr = manager.swa_pool_ptr
        block_tokens = manager.tokens_per_block
        layers = list(manager.pp_layers)
    else:
        base_ptr = manager.compress_pool_ptrs[compress_ratio]
        layers = [
            layer
            for layer in manager.pp_layers
            if manager._compress_ratios[layer] == compress_ratio
        ]
        block_tokens = manager.compressed_block_sizes[layers[0]]

    max_end_bytes = 0
    for layer in layers:
        buf = manager.get_buffers(layer, attn_type)
        offset = buf.data_ptr() - base_ptr
        assert offset >= 0 and offset % footer_scale_kv.TOKEN_BYTES == 0, (
            f"{attn_type.name} buffer for layer {layer} is not slot-aligned "
            f"to its pool base (offset {offset} bytes)"
        )
        max_end_bytes = max(max_end_bytes, offset + buf.numel() * buf.element_size())

    page_size = min(block_tokens, footer_scale_kv.PAGE_SIZE)
    page_bytes = page_size * footer_scale_kv.TOKEN_BYTES
    assert max_end_bytes % page_bytes == 0, (
        f"{attn_type.name} pool extent {max_end_bytes} is not a whole number "
        f"of {page_size}-token footer-scale pages"
    )
    pool = convert_to_torch_tensor(
        TensorWrapper(base_ptr, DataType.UINT8, (max_end_bytes // page_bytes, page_bytes))
    )
    cache[key] = pool
    return pool


class DeepseekV4FlashInferAttention(DeepseekV4TrtllmAttention):
    """DeepSeek-V4 sparse MLA served by flashinfer's SM120 packed kernels."""

    Metadata = DeepseekV4TrtllmAttentionMetadata
    kv_token_layout = "footer_scale"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, "compressor", None) is not None:
            self.compressor.enable_footer_scale_cache()

    def support_fused_rope(self) -> bool:
        # RoPE runs in mla.py; this backend receives roped q and a latent
        # cache whose rope half the model layer has already resynced.
        return False

    def _swa_append(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        latent_rows: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> None:
        """Quantize the new latent rows and scatter them into the SWA pool."""
        positions = metadata.token_positions_cuda[start_idx:end_idx]
        req_id = metadata.req_idx_per_token[start_idx:end_idx]
        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        block_table_swa = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]
        # has_fp8_kv_cache is irrelevant here: the footer-scale layout branch
        # returns its fixed token stride before any dtype sizing applies.
        token_stride = get_token_bytes(
            self.head_dim,
            self.sparse_attention_config.index_head_dim,
            self.compress_ratio,
            DeepseekV4AttentionType.SWA,
            False,
            kv_token_layout=self.kv_token_layout,
        )
        loc = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table_swa,
            swa_local_indices=positions.unsqueeze(1).contiguous(),
            swa_pool_base_ptr=metadata.sparse_mla_base_ptrs[1],
            swa_buffer_ptr=metadata.swa_buffer_ptrs[self.layer_idx],
            tokens_per_block=metadata.kv_cache_manager.tokens_per_block,
            token_stride=token_stride,
        ).view(-1)
        swa_pool = _footer_scale_pool_2d(metadata, DeepseekV4AttentionType.SWA, 1)
        footer_scale_kv.quant_scatter(swa_pool, loc, latent_rows)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ):
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if forward_args.enable_dsv4_epilogue_fusion:
            raise NotImplementedError(
                "DSv4 epilogue fusion is fused into the C++ attention op; the "
                "FlashInfer SM120 backend does not provide it."
            )
        if metadata.max_draft_tokens > 0:
            raise NotImplementedError(
                "MTP / speculative decoding is not supported by the FlashInfer "
                "DSv4 backend yet."
            )

        attn_sink = getattr(self, "attn_sink", None)
        sinks = forward_args.attention_sinks
        if sinks is None and attn_sink is not None:
            sinks = attn_sink.data
        if sinks is not None:
            sinks = sinks.to(torch.float32)

        attention_input_type = forward_args.attention_input_type
        if attention_input_type == AttentionInputType.context_only:
            start_idx, end_idx = 0, metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx, end_idx = metadata.num_ctx_tokens, metadata.num_tokens
        else:
            start_idx, end_idx = 0, metadata.num_tokens
        num_tokens = end_idx - start_idx

        output = forward_args.output
        assert output is not None, "FlashInfer DSv4 backend needs a preallocated output"
        if num_tokens == 0:
            return output

        q_view = q.view(num_tokens, self.num_heads, self.head_dim)
        if q_view.dtype != torch.bfloat16:
            raise NotImplementedError(
                "FlashInfer SM120 sparse MLA takes bf16 queries; the fused "
                "FP8-Q path must stay disabled for this backend."
            )

        # Append the new latent rows before predicting indices: each token's
        # SWA window includes itself.
        latent_cache = forward_args.latent_cache
        if latent_cache is not None and forward_args.update_kv_cache:
            self._swa_append(
                metadata, latent_cache.view(num_tokens, self.head_dim), start_idx, end_idx
            )

        swa_indices, extra_indices = self.sparse_attn_predict(
            q, k, metadata, forward_args, split_extra=True
        )
        window = self.sparse_attention_config.window_size
        positions = metadata.token_positions_cuda[start_idx:end_idx]
        swa_lens = (positions + 1).clamp(max=window).to(torch.int32)

        if self.compress_ratio > 1:
            assert extra_indices is not None
            extra_lens = (
                metadata.sparse_mla_topk_lens[self.compress_ratio][start_idx:end_idx] - window
            ).clamp(min=0)
            extra_pool = _footer_scale_pool_2d(
                metadata, DeepseekV4AttentionType.COMPRESS, self.compress_ratio
            )
            num_splits = (window + _KV_SPLIT_TILE - 1) // _KV_SPLIT_TILE + (
                extra_indices.shape[1] + _KV_SPLIT_TILE - 1
            ) // _KV_SPLIT_TILE
        else:
            extra_indices = None
            extra_lens = None
            extra_pool = None
            num_splits = (window + _KV_SPLIT_TILE - 1) // _KV_SPLIT_TILE

        out_view = output.view(num_tokens, self.num_heads, self.head_dim)
        out_lse = torch.zeros(
            num_tokens, self.num_heads, dtype=torch.float32, device=q.device
        )
        if num_tokens <= 64:
            # Split-K scratch must be initialized: per-token top-k truncation
            # can leave splits the kernel never writes, and the merge reads
            # every split slot. A -inf LSE makes an unwritten split
            # weightless; uninitialized scratch here was a decode
            # nondeterminism source.
            mid_out = torch.zeros(
                num_tokens,
                self.num_heads,
                num_splits,
                self.head_dim,
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

        sm_scale = 1.0 / (self.q_scaling * math.sqrt(self.head_dim))
        swa_pool = _footer_scale_pool_2d(metadata, DeepseekV4AttentionType.SWA, 1)
        swa_pool_paged = swa_pool.view(
            -1, footer_scale_kv.PAGE_SIZE, footer_scale_kv.TOKEN_BYTES
        )
        extra_pool_paged = None
        if extra_pool is not None:
            extra_pool_paged = extra_pool.view(
                extra_pool.shape[0], -1, footer_scale_kv.TOKEN_BYTES
            )

        _sparse_mla_op()(
            q_view.contiguous(),
            swa_pool_paged,
            swa_indices,
            out_view,
            out_lse,
            sm_scale,
            d_v=self.head_dim,
            topk_length=swa_lens,
            attn_sink=sinks,
            extra_kv_cache=extra_pool_paged,
            extra_indices=extra_indices,
            extra_topk_length=extra_lens,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )
        return output
