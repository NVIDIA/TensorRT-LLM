# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashMLA execution for DeepSeek-V4 on Hopper GPUs."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger

from . import footer_scale_kv
from .kernels import deepseek_v4_local_to_global_indices
from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import DEEPSEEK_V4_SPARSE_RATIO, DeepseekV4AttentionType

_BF16_CONTEXT_CHUNK_SIZE = 512


try:
    import tensorrt_llm.flash_mla_cpp_tllm as flash_mla_cuda
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_cuda = None
    flash_mla_sparse_fwd = None


class DeepSeekV4FlashMLA:
    """Run DeepSeek-V4 sparse attention with FlashMLA on Hopper."""

    def __init__(self, attention: TrtllmAttention, compress_ratio: int) -> None:
        if attention.mla_params is None:
            raise ValueError("DeepSeek-V4 FlashMLA requires MLA parameters")
        self._attention = attention
        self._layer_idx = attention.layer_idx
        self._compress_ratio = compress_ratio
        self._num_heads = attention.num_heads
        self._qk_nope_head_dim = attention.mla_params.qk_nope_head_dim
        self._qk_rope_head_dim = attention.mla_params.qk_rope_head_dim
        self._kv_lora_rank = attention.mla_params.kv_lora_rank
        self._v_head_dim = attention.mla_params.v_head_dim

    def _prepare_q_and_cache(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        *,
        is_generation: bool,
        rotary_cos_sin: Optional[torch.Tensor] = None,
        is_neox: bool = True,
    ) -> None:
        """Apply RoPE and append to the canonical footer-scale SWA cache."""
        num_tokens = q.shape[0]
        if num_tokens == 0:
            return
        start_idx = metadata.num_ctx_tokens if is_generation else 0
        end_idx = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
        if rotary_cos_sin is None:
            raise RuntimeError("DeepSeek-V4 Hopper requires a rotary embedding table")

        head_dim = self._kv_lora_rank + self._qk_rope_head_dim
        latent_rows = latent_cache.view(num_tokens, head_dim)
        with nvtx_range_debug("deepseek_v4_hopper_footer_scale_cache_append"):
            footer_scale_kv.apply_rope_and_append_swa(
                self._attention,
                metadata,
                q,
                latent_rows,
                start_idx,
                end_idx,
                rotary_cos_sin,
                is_neox,
                page_size=metadata.kv_cache_manager.tokens_per_block,
            )

    def _prepare_pool_indices(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        topk_indices: Optional[torch.Tensor],
        *,
        is_generation: bool,
    ) -> tuple[torch.Tensor, int, int]:
        """Convert local dual-pool indices to layer-pool-relative indices."""
        kv_cache_manager = metadata.kv_cache_manager
        window_size = metadata.window_size
        start_idx = metadata.num_ctx_tokens if is_generation else 0
        end_idx = metadata.num_tokens if is_generation else metadata.num_ctx_tokens

        req_id = metadata.req_idx_per_token[start_idx:end_idx]
        swa_local_indices = metadata.swa_local_indices_cuda[start_idx:end_idx]
        local_layer_idx = kv_cache_manager.layer_offsets[self._layer_idx]
        block_table_swa = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]
        swa_buffer_ptr = metadata.swa_buffer_ptrs[self._layer_idx]

        block_table_compressed = None
        compressed_local_indices = None
        compressed_buffer_ptr = 0
        if self._compress_ratio > 1:
            compressed_buffer_ptr = metadata.compressed_buffer_ptrs[self._layer_idx]
            block_table_compressed = metadata.compress_block_tables[self._compress_ratio]
            if self._compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
                if topk_indices is None:
                    raise ValueError("DeepSeek-V4 ratio-4 attention requires top-k indices")
                compressed_local_indices = topk_indices
            else:
                compressed_local_indices = metadata.compressed_local_indices_cuda[start_idx:end_idx]

        # Equal base/layer pointers make both output slices relative to the
        # tensors passed to FlashMLA instead of the shared sparse arena.
        global_indices = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table_swa,
            swa_local_indices=swa_local_indices,
            swa_pool_base_ptr=swa_buffer_ptr,
            swa_buffer_ptr=swa_buffer_ptr,
            tokens_per_block=kv_cache_manager.tokens_per_block,
            token_stride=footer_scale_kv.TOKEN_BYTES,
            block_table_compressed=block_table_compressed,
            compressed_local_indices=compressed_local_indices,
            compress_pool_base_ptr=compressed_buffer_ptr,
            compressed_buffer_ptr=compressed_buffer_ptr,
            compress_ratio=self._compress_ratio,
            num_compressed_indices=metadata.max_compressed_indices[self._compress_ratio],
        )
        return global_indices, self._compress_ratio, window_size

    @staticmethod
    def _pad_sparse_indices(indices: torch.Tensor, alignment: int) -> torch.Tensor:
        aligned_topk = ((indices.shape[-1] + alignment - 1) // alignment) * alignment
        if aligned_topk == indices.shape[-1]:
            return indices
        padding = indices.new_full((*indices.shape[:-1], aligned_topk - indices.shape[-1]), -1)
        return torch.cat([indices, padding], dim=-1)

    def _attention_sink(self, padded_heads: int) -> Optional[torch.Tensor]:
        attn_sink = getattr(self._attention, "attn_sink", None)
        if attn_sink is None:
            return None
        sink = attn_sink.data
        if sink.shape[0] == padded_heads:
            return sink
        if sink.shape[0] > padded_heads:
            raise ValueError(
                f"DeepSeek-V4 attention sink has {sink.shape[0]} heads, "
                f"but the Hopper path has {padded_heads}"
            )
        return torch.cat([sink, sink.new_zeros(padded_heads - sink.shape[0])])

    @staticmethod
    def _pad_decode_query(q: torch.Tensor) -> torch.Tensor:
        """Pad query heads to a shape supported by FlashMLA Hopper decode."""
        num_heads = q.shape[-2]
        padded_heads = 64 if num_heads <= 64 else 128
        if num_heads == padded_heads:
            return q

        logger.warning_once(
            f"Padding num_heads from {num_heads} to {padded_heads} "
            "for the Hopper FlashMLA sparse decode kernel",
            key="deepseek_v4_sparse_decode_hopper_padding",
        )
        padded_q = q.new_zeros((*q.shape[:-2], padded_heads, q.shape[-1]))
        padded_q[..., :num_heads, :] = q
        return padded_q

    @staticmethod
    def _prepare_bf16_pool(
        pool: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize selected footer-scale cache entries for context attention."""
        assert pool.dtype == torch.uint8 and pool.shape[-1] == footer_scale_kv.TOKEN_BYTES
        valid = indices >= 0
        safe_indices = indices.clamp_min(0).to(torch.long)
        selected_pool = footer_scale_kv.dequant_gather(
            pool,
            safe_indices.reshape(-1),
            page_size=pool.shape[1],
        ).reshape(
            -1,
            1,
            footer_scale_kv.DIM_NOPE + footer_scale_kv.DIM_ROPE,
        )

        selected_indices = torch.arange(
            indices.numel(), dtype=torch.int32, device=indices.device
        ).view_as(indices)
        selected_indices.masked_fill_(~valid, -1)
        return selected_pool, selected_indices

    @staticmethod
    def _merge_pools(
        pool_outputs: list[torch.Tensor],
        pool_lses: list[torch.Tensor],
        attention_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Merge independently normalized pools and apply one global sink."""
        if len(pool_outputs) != len(pool_lses) or not pool_outputs:
            raise ValueError("DeepSeek-V4 Hopper attention requires matching non-empty pools")

        finite_pool_lses = [
            torch.where(torch.isfinite(pool_lse), pool_lse, -torch.inf) for pool_lse in pool_lses
        ]
        max_lse = torch.stack(finite_pool_lses).amax(dim=0)
        has_valid_pool = torch.isfinite(max_lse)
        safe_max_lse = torch.where(has_valid_pool, max_lse, torch.zeros_like(max_lse))
        pool_weights = [torch.exp(pool_lse - safe_max_lse) for pool_lse in finite_pool_lses]
        denominator = torch.stack(pool_weights).sum(dim=0)
        safe_denominator = torch.where(has_valid_pool, denominator, torch.ones_like(denominator))
        output = sum(
            torch.where(
                torch.isfinite(pool_lse).unsqueeze(-1),
                pool_output,
                torch.zeros_like(pool_output),
            )
            * pool_weight.unsqueeze(-1)
            for pool_output, pool_lse, pool_weight in zip(
                pool_outputs, pool_lses, pool_weights, strict=True
            )
        ) / safe_denominator.unsqueeze(-1)

        if attention_sink is not None:
            merged_lse = torch.where(
                has_valid_pool,
                safe_max_lse + torch.log(safe_denominator),
                -torch.inf,
            )
            output *= torch.sigmoid(merged_lse - attention_sink.unsqueeze(0)).unsqueeze(-1)
        return output

    def forward_context(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        output: torch.Tensor,
        topk_indices: Optional[torch.Tensor],
        softmax_scale: float,
        rotary_cos_sin: Optional[torch.Tensor] = None,
        is_neox: bool = True,
    ) -> torch.Tensor:
        """Run the Hopper context path with FlashMLA."""
        return self._forward_bf16(
            q,
            latent_cache,
            metadata,
            output,
            topk_indices,
            softmax_scale,
            is_generation=False,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )

    @nvtx_range("forward_sparse_mla_deepseek_v4_hopper_bf16")
    def _forward_bf16(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        output: torch.Tensor,
        topk_indices: Optional[torch.Tensor],
        softmax_scale: float,
        *,
        is_generation: bool,
        rotary_cos_sin: Optional[torch.Tensor] = None,
        is_neox: bool = True,
    ) -> torch.Tensor:
        """Run dual-pool sparse MLA on Hopper using BF16 prefill kernels."""
        self._prepare_q_and_cache(
            q,
            latent_cache,
            metadata,
            is_generation=is_generation,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )
        num_tokens = q.shape[0]
        q_head_dim = self._qk_nope_head_dim + self._qk_rope_head_dim
        q_concat = q.view(num_tokens, self._num_heads, q_head_dim)
        padded_heads = ((self._num_heads + 63) // 64) * 64
        if self._num_heads != padded_heads:
            logger.warning_once(
                f"Padding num_heads from {self._num_heads} to {padded_heads} "
                "for the Hopper FlashMLA sparse attention kernel",
                key="deepseek_v4_sparse_mla_hopper_padding",
            )
            q_padded = q_concat.new_zeros((num_tokens, padded_heads, q_head_dim))
            q_padded[:, : self._num_heads] = q_concat
            q_concat = q_padded

        global_indices, compress_ratio, window_size = self._prepare_pool_indices(
            metadata, topk_indices, is_generation=is_generation
        )
        kv_cache_manager = metadata.kv_cache_manager
        swa_pool = kv_cache_manager.get_buffers(self._layer_idx, DeepseekV4AttentionType.SWA)
        compressed_pool = None
        if compress_ratio > 1:
            compressed_pool = kv_cache_manager.get_buffers(
                self._layer_idx, DeepseekV4AttentionType.COMPRESS
            )
        if flash_mla_sparse_fwd is None:
            raise RuntimeError(
                "flash_mla_sparse_fwd is unavailable; build TensorRT-LLM with FlashMLA"
            )
        attention_sink = self._attention_sink(padded_heads)
        for chunk_start in range(0, num_tokens, _BF16_CONTEXT_CHUNK_SIZE):
            chunk_end = min(chunk_start + _BF16_CONTEXT_CHUNK_SIZE, num_tokens)
            chunk_tokens = chunk_end - chunk_start
            chunk_indices = global_indices[chunk_start:chunk_end]
            chunk_q = q_concat[chunk_start:chunk_end]

            swa_pool_flat, swa_indices = self._prepare_bf16_pool(
                swa_pool, chunk_indices[:, :window_size]
            )
            swa_indices = self._pad_sparse_indices(swa_indices, alignment=128).view(
                chunk_tokens, 1, -1
            )
            swa_out, _, swa_lse = flash_mla_sparse_fwd(
                chunk_q,
                swa_pool_flat,
                swa_indices,
                softmax_scale,
                d_v=self._v_head_dim,
            )
            pool_outputs = [swa_out]
            pool_lses = [swa_lse]
            if compressed_pool is not None:
                compressed_pool_flat, compressed_indices = self._prepare_bf16_pool(
                    compressed_pool,
                    chunk_indices[:, window_size:],
                )
                compressed_indices = self._pad_sparse_indices(
                    compressed_indices, alignment=128
                ).view(chunk_tokens, 1, -1)
                compressed_out, _, compressed_lse = flash_mla_sparse_fwd(
                    chunk_q,
                    compressed_pool_flat,
                    compressed_indices,
                    softmax_scale,
                    d_v=self._v_head_dim,
                )
                pool_outputs.append(compressed_out)
                pool_lses.append(compressed_lse)

            with nvtx_range_debug("merge_deepseek_v4_hopper_attention_pools"):
                attn_out = self._merge_pools(pool_outputs, pool_lses, attention_sink)

            attn_out = attn_out[:, : self._num_heads, : self._v_head_dim]
            if self._num_heads != padded_heads:
                attn_out = attn_out.contiguous()
            output[chunk_start:chunk_end].copy_(attn_out.reshape(chunk_tokens, -1))
        return output

    @nvtx_range("forward_sparse_decode_deepseek_v4_hopper_fp8")
    def _forward_fp8(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        output: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        rotary_cos_sin: Optional[torch.Tensor] = None,
        is_neox: bool = True,
    ) -> torch.Tensor:
        """Run DeepSeek-V4 Hopper decode with FlashMLA's MODEL1 kernel."""
        if self._num_heads > 128:
            raise ValueError(
                "FlashMLA Hopper sparse decode supports at most 128 query heads, "
                f"got {self._num_heads}"
            )

        self._prepare_q_and_cache(
            q,
            latent_cache,
            metadata,
            is_generation=True,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )

        kv_cache_manager = metadata.kv_cache_manager
        swa_fp8 = kv_cache_manager.get_buffers(
            self._layer_idx, DeepseekV4AttentionType.SWA
        ).unsqueeze(2)
        compressed_fp8 = None
        if self._compress_ratio > 1:
            compressed_fp8 = kv_cache_manager.get_buffers(
                self._layer_idx, DeepseekV4AttentionType.COMPRESS
            ).unsqueeze(2)

        global_indices, compress_ratio, window_size = self._prepare_pool_indices(
            metadata, topk_indices, is_generation=True
        )
        swa_indices = self._pad_sparse_indices(
            global_indices[:, :window_size].unsqueeze(1), alignment=64
        )

        compressed_indices = None
        if compress_ratio > 1:
            if compressed_fp8 is None:
                raise RuntimeError("DeepSeek-V4 compressed FlashMLA cache is not initialized")
            compressed_indices = self._pad_sparse_indices(
                global_indices[:, window_size:].unsqueeze(1), alignment=64
            )

        num_generation_tokens = q.shape[0]
        q_head_dim = self._qk_nope_head_dim + self._qk_rope_head_dim
        q_decode = q.view(num_generation_tokens, 1, self._num_heads, q_head_dim)
        q_decode = self._pad_decode_query(q_decode)
        attention_sink = self._attention_sink(q_decode.shape[-2])

        if flash_mla_cuda is None:
            raise RuntimeError(
                "FlashMLA sparse decode is unavailable; build TensorRT-LLM with FlashMLA"
            )
        with nvtx_range_debug("flash_mla_sparse_decode_deepseek_v4_hopper"):
            if compressed_fp8 is not None:
                decode_out, _, _, _ = flash_mla_cuda.sparse_decode_fwd(
                    q_decode,
                    compressed_fp8,
                    compressed_indices,
                    None,
                    attention_sink,
                    None,
                    None,
                    swa_fp8,
                    swa_indices,
                    None,
                    self._v_head_dim,
                    softmax_scale,
                )
            else:
                decode_out, _, _, _ = flash_mla_cuda.sparse_decode_fwd(
                    q_decode,
                    swa_fp8,
                    swa_indices,
                    None,
                    attention_sink,
                    None,
                    None,
                    None,
                    None,
                    None,
                    self._v_head_dim,
                    softmax_scale,
                )

        output.copy_(
            decode_out.squeeze(1)[:, : self._num_heads, : self._v_head_dim].reshape(
                num_generation_tokens, -1
            )
        )
        return output

    def forward_generation(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        output: torch.Tensor,
        topk_indices: Optional[torch.Tensor],
        softmax_scale: float,
        rotary_cos_sin: Optional[torch.Tensor] = None,
        is_neox: bool = True,
    ) -> torch.Tensor:
        """Dispatch Hopper generation to MODEL1 decode or the BF16 fallback."""
        if flash_mla_cuda is not None and topk_indices is not None:
            return self._forward_fp8(
                q,
                latent_cache,
                metadata,
                output,
                topk_indices,
                softmax_scale,
                rotary_cos_sin,
                is_neox,
            )
        return self._forward_bf16(
            q,
            latent_cache,
            metadata,
            output,
            topk_indices,
            softmax_scale,
            is_generation=True,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )
