# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashMLA execution for DeepSeek-V4 on Hopper GPUs."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.utils import maybe_compile
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger

from .cache_manager import get_token_bytes
from .kernels import deepseek_v4_local_to_global_indices
from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import (
    DEEPSEEK_V4_FLASH_MLA_BYTES_PER_TOKEN,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
)

_FP8_E4M3_MAX = 448.0
_BF16_CONTEXT_CHUNK_SIZE = 512


@maybe_compile(dynamic=True, options={"max-autotune": True})
def _write_fp8_shadow_torch(
    pool: torch.Tensor,
    shadow: torch.Tensor,
    source_block: torch.Tensor,
    shadow_block: torch.Tensor,
    token_offset: torch.Tensor,
    is_fp8_pool: bool,
    kv_scale: torch.Tensor,
    tokens_per_block: int,
) -> None:
    """Convert cache entries to MODEL1 layout and update their shadow slots."""
    nope_dim = 448
    rope_dim = 64
    quant_block = 64
    num_scales = nope_dim // quant_block
    data_bytes = nope_dim + rope_dim * 2

    token_data = pool[source_block, token_offset]
    if is_fp8_pool:
        token_bf16 = (token_data.view(torch.float8_e4m3fn).to(torch.bfloat16) * kv_scale).to(
            torch.bfloat16
        )
    else:
        token_bf16 = token_data

    num_slots = token_bf16.shape[0]
    nope = token_bf16[:, :nope_dim].float()
    rope = token_bf16[:, nope_dim:]
    nope_blocked = nope.reshape(num_slots, num_scales, quant_block)
    block_max = nope_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale_log2 = torch.log2((block_max / _FP8_E4M3_MAX).clamp(min=1e-4)).ceil()
    scale = torch.exp2(scale_log2)
    nope_fp8 = (nope_blocked / scale).reshape(num_slots, nope_dim).to(torch.float8_e4m3fn)

    scale_e8m0 = (scale_log2.squeeze(-1) + 127).clamp(0, 255).byte()
    scale_padding = torch.nn.functional.pad(scale_e8m0, (0, 1))
    encoded_data = torch.cat(
        (
            nope_fp8.view(torch.uint8).reshape(num_slots, nope_dim),
            rope.contiguous().view(torch.uint8).reshape(num_slots, rope_dim * 2),
        ),
        dim=1,
    )

    row_stride = shadow.shape[1]
    row_base = shadow_block * row_stride
    data_start = row_base + token_offset * data_bytes
    data_indices = (
        data_start.unsqueeze(1)
        + torch.arange(data_bytes, device=pool.device, dtype=torch.long).unsqueeze(0)
    ).reshape(-1)
    shadow_flat = shadow.view(-1)
    shadow_flat[data_indices] = encoded_data.reshape(-1)

    scale_start = row_base + tokens_per_block * data_bytes + token_offset * 8
    scale_indices = (
        scale_start.unsqueeze(1)
        + torch.arange(8, device=pool.device, dtype=torch.long).unsqueeze(0)
    ).reshape(-1)
    shadow_flat[scale_indices] = scale_padding.reshape(-1)


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
        position_ids: Optional[torch.Tensor] = None,
        rotary_cos_sin: Optional[torch.Tensor] = None,
        is_neox: bool = True,
    ) -> None:
        """Apply RoPE and update the DeepSeek-V4 paged cache on Hopper."""
        if not is_generation:
            if metadata.max_ctx_seq_len > 0:
                with nvtx_range_debug("mla_rope_append_paged_kv_assign_q_deepseek_v4_hopper"):
                    self._attention.mla_rope_append_paged_kv_assign_q(
                        q, latent_cache, metadata, is_generation=False
                    )
            return

        num_tokens = q.shape[0]
        q_head_dim = self._qk_nope_head_dim + self._qk_rope_head_dim
        q_view = q.view(-1, self._num_heads, q_head_dim)
        q_pe = q_view[..., self._qk_nope_head_dim :]
        num_seqs = metadata.kv_lens_cuda_runtime.size(0)
        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if self._attention.has_fp8_kv_cache:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
            quant_q_buffer = torch.empty(
                num_tokens,
                self._num_heads,
                self._kv_lora_rank + self._qk_rope_head_dim,
                dtype=torch.uint8,
                device=q.device,
            )

        with nvtx_range_debug("mla_rope_generation_deepseek_v4_hopper"):
            self._attention.mla_rope_generation(
                q,
                q_pe,
                latent_cache,
                metadata,
                cu_q_seqlens,
                cu_kv_seqlens,
                fmha_scheduler_counter,
                mla_bmm1_scale,
                mla_bmm2_scale,
                quant_q_buffer,
            )

        # The FP8 generation kernel writes the rotated/quantized query to its
        # auxiliary buffer. FlashMLA consumes the BF16 query, so rotate that
        # view explicitly as well.
        if self._attention.has_fp8_kv_cache:
            if position_ids is None or position_ids.numel() != num_tokens:
                position_ids_shape = None if position_ids is None else tuple(position_ids.shape)
                raise ValueError(
                    "Expected one position ID per generation token, got "
                    f"{position_ids_shape=} for {num_tokens=}"
                )
            if rotary_cos_sin is None:
                raise RuntimeError(
                    "DeepSeek-V4 Hopper FP8 decode requires a rotary embedding table"
                )
            torch.ops.trtllm.mla_rope_inplace(
                q_view,
                position_ids.view(-1).contiguous(),
                rotary_cos_sin,
                self._num_heads,
                self._qk_nope_head_dim,
                self._qk_rope_head_dim,
                False,
                is_neox,
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
        token_stride = get_token_bytes(
            kv_cache_manager.head_dim,
            metadata.indexer_head_dim,
            self._compress_ratio,
            DeepseekV4AttentionType.SWA,
            False,
        )
        global_indices = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table_swa,
            swa_local_indices=swa_local_indices,
            swa_pool_base_ptr=swa_buffer_ptr,
            swa_buffer_ptr=swa_buffer_ptr,
            tokens_per_block=kv_cache_manager.tokens_per_block,
            token_stride=token_stride,
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
        head_dim: int,
        kv_scale: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantize only the FP8 cache entries selected by sparse attention."""
        pool_flat = pool.reshape(-1, 1, head_dim)
        if pool.dtype == torch.bfloat16:
            return pool_flat, indices

        valid = indices >= 0
        safe_indices = indices.clamp_min(0).to(torch.long)
        selected_pool = pool_flat.view(torch.float8_e4m3fn)[safe_indices.reshape(-1)].to(
            torch.bfloat16
        )
        selected_pool *= kv_scale

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
        position_ids: Optional[torch.Tensor] = None,
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
            position_ids=position_ids,
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
        position_ids: Optional[torch.Tensor] = None,
        rotary_cos_sin: Optional[torch.Tensor] = None,
        is_neox: bool = True,
    ) -> torch.Tensor:
        """Run dual-pool sparse MLA on Hopper using BF16 prefill kernels."""
        self._prepare_q_and_cache(
            q,
            latent_cache,
            metadata,
            is_generation=is_generation,
            position_ids=position_ids,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )
        if self._compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
            self._update_fp8_shadows(
                metadata,
                position_ids,
                is_generation=is_generation,
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
        head_dim = kv_cache_manager.head_dim
        kv_scale = self._attention.kv_scale_quant_orig
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
                swa_pool, chunk_indices[:, :window_size], head_dim, kv_scale
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
                    head_dim,
                    kv_scale,
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
        position_ids: Optional[torch.Tensor] = None,
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
            position_ids=position_ids,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )

        swa_fp8, compressed_fp8 = self._update_fp8_shadows(
            metadata,
            position_ids,
            is_generation=True,
        )

        global_indices, compress_ratio, window_size = self._prepare_pool_indices(
            metadata, topk_indices, is_generation=True
        )
        kv_cache_manager = metadata.kv_cache_manager
        global_indices[:, :window_size] = kv_cache_manager.map_flash_mla_shadow_token_indices(
            self._layer_idx,
            DeepseekV4AttentionType.SWA,
            global_indices[:, :window_size],
        )
        swa_indices = self._pad_sparse_indices(
            global_indices[:, :window_size].unsqueeze(1), alignment=64
        )

        compressed_indices = None
        if compress_ratio > 1:
            if compressed_fp8 is None:
                raise RuntimeError("DeepSeek-V4 compressed FlashMLA cache is not initialized")
            global_indices[:, window_size:] = kv_cache_manager.map_flash_mla_shadow_token_indices(
                self._layer_idx,
                DeepseekV4AttentionType.COMPRESS,
                global_indices[:, window_size:],
            )
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
        position_ids: Optional[torch.Tensor] = None,
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
                position_ids,
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
            position_ids=position_ids,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )

    def _update_fp8_shadows(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        position_ids: Optional[torch.Tensor],
        *,
        is_generation: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Write newly appended cache entries through to FlashMLA's format."""
        kv_cache_manager = metadata.kv_cache_manager
        swa_pool = kv_cache_manager.get_buffers(self._layer_idx, DeepseekV4AttentionType.SWA)
        swa_fp8 = self._update_shadow_pool(
            swa_pool,
            metadata,
            DeepseekV4AttentionType.SWA,
            compress_ratio=1,
            position_ids=position_ids,
            is_generation=is_generation,
        )

        compressed_fp8 = None
        if self._compress_ratio > 1:
            compressed_pool = kv_cache_manager.get_buffers(
                self._layer_idx, DeepseekV4AttentionType.COMPRESS
            )
            compressed_fp8 = self._update_shadow_pool(
                compressed_pool,
                metadata,
                DeepseekV4AttentionType.COMPRESS,
                compress_ratio=self._compress_ratio,
                position_ids=None,
                is_generation=is_generation,
            )
        return swa_fp8, compressed_fp8

    def _update_shadow_pool(
        self,
        pool_raw: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        attn_type: DeepseekV4AttentionType,
        compress_ratio: int,
        position_ids: Optional[torch.Tensor],
        *,
        is_generation: bool,
    ) -> torch.Tensor:
        """Update a persistent MODEL1 view for the entries written this pass."""
        # TODO: Move the shadow update into the canonical cache
        # append kernels so MODEL1 does not need a separate write-through pass.
        pool = pool_raw
        is_fp8_pool = pool.dtype != torch.bfloat16
        kv_scale = self._attention.kv_scale_quant_orig

        _, tokens_per_block, head_dim = pool.shape
        nope_dim = 448
        rope_dim = 64
        bytes_per_token = DEEPSEEK_V4_FLASH_MLA_BYTES_PER_TOKEN
        if head_dim != nope_dim + rope_dim:
            raise ValueError(
                f"DeepSeek-V4 MODEL1 expects head_dim={nope_dim + rope_dim}, got {head_dim}"
            )

        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self._layer_idx]
        if attn_type == DeepseekV4AttentionType.SWA:
            block_table = metadata.sliding_block_tables[
                local_layer_idx, DeepseekV4AttentionType.SWA.value
            ]
        elif attn_type == DeepseekV4AttentionType.COMPRESS:
            block_table = metadata.compress_block_tables[compress_ratio]
        else:
            raise ValueError(f"Unsupported DeepSeek-V4 FP8 shadow type: {attn_type}")

        shadow = metadata.kv_cache_manager.get_flash_mla_shadow_buffer(self._layer_idx, attn_type)
        shadow_num_blocks = shadow.shape[0]
        shadow_view = shadow.view(
            shadow_num_blocks,
            tokens_per_block,
            1,
            bytes_per_token,
        )
        if block_table.shape[1] == 0:
            return shadow_view

        with nvtx_range_debug("deepseek_v4_fp8_shadow_update"):
            source_block, shadow_block, token_offset = self._get_shadow_write_slots(
                metadata,
                block_table,
                attn_type,
                compress_ratio,
                position_ids,
                is_generation=is_generation,
            )
            self._write_fp8_shadow(
                pool,
                shadow,
                source_block,
                shadow_block,
                token_offset,
                is_fp8_pool,
                kv_scale,
                tokens_per_block,
            )

        return shadow_view

    def _get_shadow_write_slots(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        block_table: torch.Tensor,
        attn_type: DeepseekV4AttentionType,
        compress_ratio: int,
        position_ids: Optional[torch.Tensor],
        *,
        is_generation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map cache entries written in this phase to their physical slots."""
        device = block_table.device
        empty = torch.empty(0, dtype=torch.long, device=device)
        request_start = metadata.num_contexts if is_generation else 0
        num_requests = metadata.num_generations if is_generation else metadata.num_contexts
        request_end = request_start + num_requests
        if num_requests == 0:
            return empty, empty, empty

        if attn_type == DeepseekV4AttentionType.SWA:
            token_start = metadata.num_ctx_tokens if is_generation else 0
            token_end = metadata.num_tokens if is_generation else metadata.num_ctx_tokens
            request_grid = metadata.req_idx_per_token[token_start:token_end].to(torch.long)
            if position_ids is None or position_ids.numel() != request_grid.numel():
                position_ids_shape = None if position_ids is None else tuple(position_ids.shape)
                raise ValueError(
                    "Expected one position ID per DeepSeek-V4 cache write, got "
                    f"{position_ids_shape=} for {request_grid.numel()} tokens"
                )
            token_position = position_ids.reshape(-1).to(torch.long)
            valid = (
                (request_grid >= request_start)
                & (request_grid < request_end)
                & (token_position >= 0)
            )
            request_grid = request_grid.clamp(min=request_start, max=request_end - 1)
            if not is_generation:
                # Context tokens outside the final SWA window may use scratch
                # slots, which are not persistent cache pages.
                min_live_position = (
                    metadata.kv_lens_cuda_runtime[request_grid].to(torch.long)
                    - metadata.window_size
                )
                valid &= token_position >= min_live_position
        elif attn_type == DeepseekV4AttentionType.COMPRESS:
            max_new_tokens = (
                (metadata.num_gen_tokens_per_seq + compress_ratio - 1) // compress_ratio
                if is_generation
                else metadata.max_ctx_compressed_tokens[compress_ratio]
            )
            if max_new_tokens == 0:
                return empty, empty, empty
            all_slots = torch.arange(num_requests * max_new_tokens, device=device, dtype=torch.long)
            request_grid = all_slots // max_new_tokens
            token_grid = all_slots % max_new_tokens
            request_grid = request_grid + request_start
            token_position = (
                metadata.past_kv_lens_cuda[compress_ratio][request_grid].to(torch.long) + token_grid
            )
            valid = token_grid < metadata.new_comp_kv_lens_cuda[compress_ratio][request_grid].to(
                torch.long
            )
        else:
            raise ValueError(f"Unsupported DeepSeek-V4 FlashMLA cache type: {attn_type}")

        return metadata.kv_cache_manager.map_flash_mla_shadow_write_slots(
            self._layer_idx,
            attn_type,
            block_table,
            request_grid,
            token_position,
            valid,
        )

    def _write_fp8_shadow(
        self,
        pool: torch.Tensor,
        shadow: torch.Tensor,
        source_block: torch.Tensor,
        shadow_block: torch.Tensor,
        token_offset: torch.Tensor,
        is_fp8_pool: bool,
        kv_scale: torch.Tensor,
        tokens_per_block: int,
    ) -> None:
        if source_block.numel() == 0:
            return

        # TODO: Fuse MODEL1 quantization and layout conversion
        # into the compressor and MLA cache-append kernels.
        _write_fp8_shadow_torch(
            pool,
            shadow,
            source_block,
            shadow_block,
            token_offset,
            is_fp8_pool,
            kv_scale,
            tokens_per_block,
        )
