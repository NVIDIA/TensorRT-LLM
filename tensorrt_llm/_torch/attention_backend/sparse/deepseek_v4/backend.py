# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._utils import nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

from .cache_manager import get_token_bytes
from .compressor import Compressor
from .indexer import DeepseekV4Indexer
from .kernels import deepseek_v4_local_to_global_indices
from .metadata import DeepseekV4TrtllmAttentionMetadata
from .params import DEEPSEEK_V4_SPARSE_RATIO, DeepseekV4AttentionType, DeepSeekV4Params

try:
    import tensorrt_llm.flash_mla_cpp_tllm as flash_mla_cuda
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_cuda = None
    flash_mla_sparse_fwd = None

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import SparseAttentionConfig


class DeepseekV4TrtllmAttention(TrtllmAttention):
    Metadata = DeepseekV4TrtllmAttentionMetadata

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
        sparse_attention_config: Optional["SparseAttentionConfig"] = None,
        sparse_params: Optional[DeepSeekV4Params] = None,
        dtype: Optional[torch.dtype] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        **kwargs,
    ):
        if sparse_attention_config is None:
            sparse_attention_config = sparse_params
        assert sparse_attention_config is not None, (
            "sparse_attention_config is required for DeepseekV4TrtllmAttention and cannot be None"
        )
        if sparse_params is None:
            sparse_params = sparse_attention_config.to_sparse_params()
        assert sparse_params is not None, (
            "sparse_params is required for DeepseekV4TrtllmAttention and cannot be None"
        )
        kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
        self.use_fp8_ds_mla = kv_cache_dtype == "fp8_ds_mla"
        assert mla_params is not None, "DeepSeek-V4 attention requires MLA parameters"
        mla_params = replace(
            mla_params,
            v_head_dim=head_dim,
            rope_append=False,
        )
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

        self.sparse_attention_config = sparse_attention_config
        self.compress_ratio = sparse_attention_config.compress_ratios[layer_idx]

        if self.compress_ratio == 4:
            self.indexer = DeepseekV4Indexer(
                quant_config,
                pos_embd_params,
                mla_params,
                skip_create_weights_in_init,
                sparse_params,
                dtype,
                self.compress_ratio,
                layer_idx,
                aux_stream,
            )

        if self.compress_ratio > 1:
            rms_norm_eps = 1e-6
            has_fp8_kv_cache = False
            if quant_config is not None:
                has_fp8_kv_cache = quant_config.layer_quant_mode.has_fp8_kv_cache()
            kv_cache_dtype = "fp8_pertensor" if has_fp8_kv_cache else "default"
            self.compressor = Compressor(
                mla_params,
                layer_idx,
                self.compress_ratio,
                rms_norm_eps,
                skip_create_weights_in_init,
                pos_embd_params,
                kv_cache_dtype=kv_cache_dtype,
                dtype=dtype,
                rotate_activation=False,
            )
            if self.use_fp8_ds_mla:
                self.compressor.enable_footer_scale_cache()

    def _prepare_sparse_forward_args(
        self,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> None:
        attention_input_type = forward_args.attention_input_type
        if attention_input_type == AttentionInputType.context_only:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
        else:
            start_idx = 0
            end_idx = metadata.num_tokens

        sparse_args = forward_args.sparse_runtime_params
        sparse_args.sparse_attn_kv_lens = metadata.sparse_mla_topk_lens[self.compress_ratio][
            start_idx:end_idx
        ]
        if self.compress_ratio > 1:
            sparse_args.aux_kv_cache_pool_ptr = metadata.sparse_mla_base_ptrs[self.compress_ratio]
        else:
            sparse_args.aux_kv_cache_pool_ptr = None

        metadata.num_sparse_topk = (
            self.sparse_attention_config.window_size
            + metadata.max_compressed_indices[self.compress_ratio]
        )

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
        attn_sink = getattr(self, "attn_sink", None)
        if attn_sink is not None:
            if forward_args.attention_sinks is None:
                forward_args = replace(forward_args, attention_sinks=attn_sink.data)

        self._prepare_sparse_forward_args(metadata, forward_args)
        return super().forward(q, k, v, metadata, forward_args=forward_args)

    def _unit_scale(self, like: torch.Tensor) -> torch.Tensor:
        """Cached [1.0], for scale pointers the Triton kernel cannot take as null."""
        cached = getattr(self, "_unit_scale_tensor", None)
        if cached is None or cached.device != like.device:
            cached = torch.ones(1, dtype=torch.float32, device=like.device)
            self._unit_scale_tensor = cached
        return cached

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert local indices (SWA + compressed) to global pool indices."""
        layer_idx = self.layer_idx
        kv_cache_manager = metadata.kv_cache_manager
        attention_input_type = forward_args.attention_input_type

        swa_pool_base_ptr = metadata.sparse_mla_base_ptrs[1]

        # Get cached buffer pointers
        swa_buffer_ptr = metadata.swa_buffer_ptrs[layer_idx]

        # Token stride
        index_head_dim = self.sparse_attention_config.index_head_dim
        has_fp8_kv_cache = False
        if self.quant_config is not None:
            has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache()
        token_stride = get_token_bytes(
            self.head_dim,
            index_head_dim,
            self.compress_ratio,
            DeepseekV4AttentionType.SWA,
            has_fp8_kv_cache,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
        )

        # Select token range based on phase
        if attention_input_type == AttentionInputType.context_only:
            start_idx = 0
            end_idx = metadata.num_ctx_tokens
        elif attention_input_type == AttentionInputType.generation_only:
            start_idx = metadata.num_ctx_tokens
            end_idx = metadata.num_tokens
        else:
            start_idx = 0
            end_idx = metadata.num_tokens

        # Use global req_id directly
        req_id = metadata.req_idx_per_token[start_idx:end_idx]
        swa_local_indices = metadata.swa_local_indices_cuda[start_idx:end_idx]
        local_layer_idx = kv_cache_manager.layer_offsets[layer_idx]
        block_table_swa = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]

        if self.compress_ratio > 1:
            compressed_buffer_ptr = metadata.compressed_buffer_ptrs[layer_idx]
            compress_pool_base_ptr = metadata.sparse_mla_base_ptrs[self.compress_ratio]
            block_table_compressed = metadata.compress_block_tables[self.compress_ratio]
            if self.compress_ratio == 4:
                sparse_backend_args = forward_args.sparse_backend_args
                assert sparse_backend_args is not None, (
                    "sparse_backend_args is required when compress_ratio=4"
                )
                topk_indices = sparse_backend_args.topk_indices
                assert topk_indices is not None, "topk_indices is required when compress_ratio=4"
                compressed_local_indices = topk_indices
            else:
                compressed_local_indices = metadata.compressed_local_indices_cuda[start_idx:end_idx]
        else:
            compressed_buffer_ptr = 0
            compress_pool_base_ptr = 0
            block_table_compressed = None
            compressed_local_indices = None

        # FMHA scheduler prologue: this kernel is the last one before FMHA, so it owns
        # the tile-counter reset and bmm scale derivation the MLA RoPE kernels used to
        # do two launches earlier. Generation only -- context uses the attention
        # workspace.
        sched_kwargs = {}
        if (
            attention_input_type == AttentionInputType.generation_only
            and has_fp8_kv_cache
            and forward_args.fmha_scheduler_counter is not None
            and forward_args.mla_bmm1_scale is not None
            and forward_args.mla_bmm2_scale is not None
        ):
            sched_kwargs = dict(
                fmha_tile_counter=forward_args.fmha_scheduler_counter,
                bmm1_scale=forward_args.mla_bmm1_scale,
                bmm2_scale=forward_args.mla_bmm2_scale,
                # Mirrors attentionOp.cpp: quant_scale_o is the attention-output
                # quant scale, and both dequant scales are the KV cache scale.
                # The Triton kernel always dereferences quant_scale_o, so an absent
                # out_scale needs an explicit 1.0 -- the value mlaKernels.cu:490
                # substitutes for a null pointer. Falling back to the KV scale here
                # would square it into bmm2.
                quant_scale_o=(
                    forward_args.out_scale
                    if forward_args.out_scale is not None
                    else self._unit_scale(self.kv_scale_quant_orig)
                ),
                dequant_scale_q=self.kv_scale_quant_orig,
                dequant_scale_kv=self.kv_scale_quant_orig,
                host_bmm1_scale=1.0
                / (
                    self.q_scaling * math.sqrt(float(self.qk_nope_head_dim + self.qk_rope_head_dim))
                ),
            )

        result = deepseek_v4_local_to_global_indices(
            req_id=req_id,
            block_table_swa=block_table_swa,
            swa_local_indices=swa_local_indices,
            swa_pool_base_ptr=swa_pool_base_ptr,
            swa_buffer_ptr=swa_buffer_ptr,
            tokens_per_block=kv_cache_manager.tokens_per_block,
            token_stride=token_stride,
            block_table_compressed=block_table_compressed,
            compressed_local_indices=compressed_local_indices,
            compress_pool_base_ptr=compress_pool_base_ptr,
            compressed_buffer_ptr=compressed_buffer_ptr,
            compress_ratio=self.compress_ratio,
            num_compressed_indices=metadata.max_compressed_indices[self.compress_ratio],
            **sched_kwargs,
            split_extra=self.use_fp8_ds_mla,
        )

        if self.use_fp8_ds_mla:
            return result
        return result, None

    def sparse_kv_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: DeepseekV4TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return None, None

    def _prepare_hopper_q_and_cache(
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
            if getattr(metadata, "max_ctx_seq_len", 0) > 0:
                with nvtx_range_debug("mla_rope_append_paged_kv_assign_q_deepseek_v4_hopper"):
                    self.mla_rope_append_paged_kv_assign_q(
                        q, latent_cache, metadata, is_generation=False
                    )
            return

        num_tokens = q.shape[0]
        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q_view = q.view(-1, self.num_heads, q_head_dim)
        q_pe = q_view[..., self.qk_nope_head_dim :]
        num_seqs = metadata.kv_lens_cuda_runtime.size(0)
        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if self.has_fp8_kv_cache:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
            quant_q_buffer = torch.empty(
                num_tokens,
                self.num_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
                dtype=torch.uint8,
                device=q.device,
            )

        with nvtx_range_debug("mla_rope_generation_deepseek_v4_hopper"):
            self.mla_rope_generation(
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
        if self.has_fp8_kv_cache:
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
                self.num_heads,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                False,
                is_neox,
            )

    def _prepare_hopper_pool_indices(
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
        local_layer_idx = kv_cache_manager.layer_offsets[self.layer_idx]
        block_table_swa = metadata.sliding_block_tables[
            local_layer_idx, DeepseekV4AttentionType.SWA.value
        ]
        swa_buffer_ptr = metadata.swa_buffer_ptrs[self.layer_idx]

        block_table_compressed = None
        compressed_local_indices = None
        compressed_buffer_ptr = 0
        if self.compress_ratio > 1:
            compressed_buffer_ptr = metadata.compressed_buffer_ptrs[self.layer_idx]
            block_table_compressed = metadata.compress_block_tables[self.compress_ratio]
            if self.compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
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
            self.compress_ratio,
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
            compress_ratio=self.compress_ratio,
            num_compressed_indices=metadata.max_compressed_indices[self.compress_ratio],
        )
        return global_indices, self.compress_ratio, window_size

    @staticmethod
    def _pad_sparse_indices(indices: torch.Tensor, alignment: int) -> torch.Tensor:
        aligned_topk = ((indices.shape[-1] + alignment - 1) // alignment) * alignment
        if aligned_topk == indices.shape[-1]:
            return indices
        padding = indices.new_full((*indices.shape[:-1], aligned_topk - indices.shape[-1]), -1)
        return torch.cat([indices, padding], dim=-1)

    def _attention_sink(self, padded_heads: int) -> Optional[torch.Tensor]:
        attn_sink = getattr(self, "attn_sink", None)
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
    def _pad_hopper_decode_query(q: torch.Tensor) -> torch.Tensor:
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
    def _prepare_hopper_bf16_pool(
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
    def _merge_hopper_pools(
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
        pool_weights = [torch.exp(pool_lse - max_lse) for pool_lse in finite_pool_lses]
        denominator = torch.stack(pool_weights).sum(dim=0)
        output = sum(
            torch.where(
                torch.isfinite(pool_lse).unsqueeze(-1),
                pool_output,
                torch.zeros_like(pool_output),
            )
            * pool_weight.unsqueeze(-1)
            for pool_output, pool_lse, pool_weight in zip(pool_outputs, pool_lses, pool_weights)
        ) / denominator.unsqueeze(-1)

        if attention_sink is not None:
            merged_lse = max_lse + torch.log(denominator)
            output *= torch.sigmoid(merged_lse - attention_sink.unsqueeze(0)).unsqueeze(-1)
        return output

    @nvtx_range("forward_sparse_mla_deepseek_v4_hopper_bf16")
    def forward_hopper_bf16(
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
        if not is_generation:
            self._fp8_shadow_needs_bulk = True

        self._prepare_hopper_q_and_cache(
            q,
            latent_cache,
            metadata,
            is_generation=is_generation,
            position_ids=position_ids,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )

        num_tokens = q.shape[0]
        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q_concat = q.view(num_tokens, self.num_heads, q_head_dim)
        padded_heads = ((self.num_heads + 63) // 64) * 64
        if self.num_heads != padded_heads:
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to {padded_heads} "
                "for the Hopper FlashMLA sparse attention kernel",
                key="deepseek_v4_sparse_mla_hopper_padding",
            )
            q_padded = q_concat.new_zeros((num_tokens, padded_heads, q_head_dim))
            q_padded[:, : self.num_heads] = q_concat
            q_concat = q_padded

        global_indices, compress_ratio, window_size = self._prepare_hopper_pool_indices(
            metadata, topk_indices, is_generation=is_generation
        )
        kv_cache_manager = metadata.kv_cache_manager
        head_dim = kv_cache_manager.head_dim
        kv_scale = self.kv_scale_quant_orig

        swa_pool = kv_cache_manager.get_buffers(self.layer_idx, DeepseekV4AttentionType.SWA)
        swa_pool_flat, swa_indices = self._prepare_hopper_bf16_pool(
            swa_pool, global_indices[:, :window_size], head_dim, kv_scale
        )
        swa_indices = self._pad_sparse_indices(swa_indices, alignment=128).view(num_tokens, 1, -1)

        if flash_mla_sparse_fwd is None:
            raise RuntimeError(
                "flash_mla_sparse_fwd is unavailable; build TensorRT-LLM with FlashMLA"
            )
        swa_out, _, swa_lse = flash_mla_sparse_fwd(
            q_concat,
            swa_pool_flat,
            swa_indices,
            softmax_scale,
            d_v=self.v_head_dim,
        )

        pool_outputs = [swa_out]
        pool_lses = [swa_lse]
        if compress_ratio > 1:
            compressed_pool = kv_cache_manager.get_buffers(
                self.layer_idx, DeepseekV4AttentionType.COMPRESS
            )
            compressed_pool_flat, compressed_indices = self._prepare_hopper_bf16_pool(
                compressed_pool,
                global_indices[:, window_size:],
                head_dim,
                kv_scale,
            )
            compressed_indices = self._pad_sparse_indices(compressed_indices, alignment=128).view(
                num_tokens, 1, -1
            )
            compressed_out, _, compressed_lse = flash_mla_sparse_fwd(
                q_concat,
                compressed_pool_flat,
                compressed_indices,
                softmax_scale,
                d_v=self.v_head_dim,
            )
            pool_outputs.append(compressed_out)
            pool_lses.append(compressed_lse)

        with nvtx_range_debug("merge_deepseek_v4_hopper_attention_pools"):
            attn_out = self._merge_hopper_pools(
                pool_outputs,
                pool_lses,
                self._attention_sink(padded_heads),
            )

        attn_out = attn_out[:, : self.num_heads, : self.v_head_dim]
        if self.num_heads != padded_heads:
            attn_out = attn_out.contiguous()
        output.copy_(attn_out.reshape(num_tokens, -1))
        return output

    @nvtx_range("forward_sparse_decode_deepseek_v4_hopper_fp8")
    def forward_hopper_fp8(
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
        if self.num_heads > 128:
            raise ValueError(
                "FlashMLA Hopper sparse decode supports at most 128 query heads, "
                f"got {self.num_heads}"
            )

        self._prepare_hopper_q_and_cache(
            q,
            latent_cache,
            metadata,
            is_generation=True,
            position_ids=position_ids,
            rotary_cos_sin=rotary_cos_sin,
            is_neox=is_neox,
        )

        kv_cache_manager = metadata.kv_cache_manager
        swa_pool = kv_cache_manager.get_buffers(self.layer_idx, DeepseekV4AttentionType.SWA)
        swa_fp8 = self._ensure_hopper_fp8_shadow_current(
            swa_pool, metadata, DeepseekV4AttentionType.SWA, compress_ratio=1
        )

        global_indices, compress_ratio, window_size = self._prepare_hopper_pool_indices(
            metadata, topk_indices, is_generation=True
        )
        swa_indices = self._pad_sparse_indices(
            global_indices[:, :window_size].unsqueeze(1), alignment=64
        )

        compressed_fp8 = None
        compressed_indices = None
        if compress_ratio > 1:
            compressed_pool = kv_cache_manager.get_buffers(
                self.layer_idx, DeepseekV4AttentionType.COMPRESS
            )
            compressed_fp8 = self._ensure_hopper_fp8_shadow_current(
                compressed_pool,
                metadata,
                DeepseekV4AttentionType.COMPRESS,
                compress_ratio,
            )
            compressed_indices = self._pad_sparse_indices(
                global_indices[:, window_size:].unsqueeze(1), alignment=64
            )

        self._fp8_shadow_needs_bulk = False
        num_generation_tokens = q.shape[0]
        q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q_decode = q.view(num_generation_tokens, 1, self.num_heads, q_head_dim)
        q_decode = self._pad_hopper_decode_query(q_decode)
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
                    self.v_head_dim,
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
                    self.v_head_dim,
                    softmax_scale,
                )

        output.copy_(
            decode_out.squeeze(1)[:, : self.num_heads, : self.v_head_dim].reshape(
                num_generation_tokens, -1
            )
        )
        return output

    def forward_hopper_generation(
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
            return self.forward_hopper_fp8(
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
        return self.forward_hopper_bf16(
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

    def _ensure_hopper_fp8_shadow_current(
        self,
        pool_raw: torch.Tensor,
        metadata: DeepseekV4TrtllmAttentionMetadata,
        attn_type: DeepseekV4AttentionType,
        compress_ratio: int,
    ) -> torch.Tensor:
        """Update a persistent MODEL1 FP8 shadow of a paged KV pool."""
        pool = pool_raw
        is_fp8_pool = pool.dtype != torch.bfloat16
        kv_scale = self.kv_scale_quant_orig

        num_blocks, tokens_per_block, head_dim = pool.shape
        nope_dim = 448
        rope_dim = 64
        quant_block = 64
        num_scales = nope_dim // quant_block
        data_bytes = nope_dim + rope_dim * 2
        bytes_per_token = data_bytes + 8
        device = pool.device
        if head_dim != nope_dim + rope_dim:
            raise ValueError(
                f"DeepSeek-V4 MODEL1 expects head_dim={nope_dim + rope_dim}, got {head_dim}"
            )

        local_layer_idx = metadata.kv_cache_manager.layer_offsets[self.layer_idx]
        if attn_type == DeepseekV4AttentionType.SWA:
            block_table = metadata.sliding_block_tables[
                local_layer_idx, DeepseekV4AttentionType.SWA.value
            ]
        elif attn_type == DeepseekV4AttentionType.COMPRESS:
            block_table = metadata.compress_block_tables[compress_ratio]
        else:
            raise ValueError(f"Unsupported DeepSeek-V4 FP8 shadow type: {attn_type}")

        kv_lens = metadata.kv_lens_cuda_runtime
        num_requests = kv_lens.shape[0]
        max_blocks_per_sequence = block_table.shape[1]
        self._ensure_hopper_shadow_state(
            attn_type,
            num_blocks,
            tokens_per_block,
            bytes_per_token,
            device,
        )

        if getattr(self, "_fp8_shadow_needs_bulk", False):
            for block_fill in self._fp8_block_fill_gpu.values():
                block_fill.zero_()
            self._fp8_shadow_needs_bulk = False

        shadow = self._fp8_shadows[attn_type]
        block_fill_gpu = self._fp8_block_fill_gpu[attn_type]
        if num_requests == 0 or max_blocks_per_sequence == 0:
            return shadow[:num_blocks].reshape(num_blocks, tokens_per_block, 1, bytes_per_token)

        with nvtx_range_debug("deepseek_v4_fp8_shadow_update"):
            effective_kv_lens = (
                kv_lens // compress_ratio
                if attn_type == DeepseekV4AttentionType.COMPRESS and compress_ratio > 1
                else kv_lens
            )
            request_grid, block_grid, token_grid = self._get_fp8_shadow_update_grid(
                num_requests, max_blocks_per_sequence, tokens_per_block, device
            )
            kv_len_per_slot = effective_kv_lens.to(torch.long)[request_grid]
            token_position = block_grid * tokens_per_block + token_grid
            physical_block = block_table.to(torch.long)[request_grid, block_grid]

            physical_valid = (physical_block >= 0) & (physical_block < num_blocks)
            physical_safe = torch.where(
                physical_valid, physical_block, torch.zeros_like(physical_block)
            )
            current_fill = block_fill_gpu[physical_safe].to(torch.long)
            valid = (
                (token_position < kv_len_per_slot)
                & (kv_len_per_slot > 0)
                & physical_valid
                & (token_grid >= current_fill)
            )

            scatter_block = torch.where(
                valid,
                physical_safe,
                torch.full_like(physical_safe, num_blocks),
            )
            gather_block = torch.where(
                physical_valid, physical_safe, torch.zeros_like(physical_safe)
            )
            token_data = pool[gather_block, token_grid]
            if is_fp8_pool:
                token_bf16 = (
                    token_data.view(torch.float8_e4m3fn).to(torch.bfloat16) * kv_scale
                ).to(torch.bfloat16)
            else:
                token_bf16 = token_data

            num_slots = token_bf16.shape[0]
            nope = token_bf16[:, :nope_dim].float()
            rope = token_bf16[:, nope_dim:]
            nope_blocked = nope.reshape(num_slots, num_scales, quant_block)
            block_max = nope_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            scale_log2 = torch.log2((block_max / 448.0).clamp(min=1e-4)).ceil()
            scale = torch.exp2(scale_log2)
            nope_fp8 = (nope_blocked / scale).reshape(num_slots, nope_dim).to(torch.float8_e4m3fn)

            scale_e8m0 = (scale_log2.squeeze(-1) + 127).clamp(0, 255).byte()
            scale_padding = self._get_fp8_scale_padding(num_slots, device)
            scale_padding.zero_()
            scale_padding[:, :num_scales] = scale_e8m0

            nope_bytes = nope_fp8.view(torch.uint8).reshape(num_slots, nope_dim)
            rope_bytes = rope.contiguous().view(torch.uint8).reshape(num_slots, rope_dim * 2)
            row_stride = tokens_per_block * bytes_per_token
            row_base = scatter_block * row_stride
            shadow_flat = shadow.view(-1)

            data_start = row_base + token_grid * data_bytes
            nope_indices = (
                data_start.unsqueeze(1)
                + self._get_fp8_shadow_offsets(nope_dim, device).unsqueeze(0)
            ).reshape(-1)
            shadow_flat[nope_indices] = nope_bytes.reshape(-1)

            rope_start = data_start + nope_dim
            rope_indices = (
                rope_start.unsqueeze(1)
                + self._get_fp8_shadow_offsets(rope_dim * 2, device).unsqueeze(0)
            ).reshape(-1)
            shadow_flat[rope_indices] = rope_bytes.reshape(-1)

            scale_start = row_base + tokens_per_block * data_bytes + token_grid * 8
            scale_indices = (
                scale_start.unsqueeze(1) + self._get_fp8_shadow_offsets(8, device).unsqueeze(0)
            ).reshape(-1)
            shadow_flat[scale_indices] = scale_padding.reshape(-1)

            update_value = torch.where(valid, token_grid + 1, torch.zeros_like(token_grid))
            block_fill_gpu.scatter_reduce_(
                0,
                physical_safe,
                update_value.to(block_fill_gpu.dtype),
                reduce="amax",
                include_self=True,
            )

        return shadow[:num_blocks].reshape(num_blocks, tokens_per_block, 1, bytes_per_token)

    def _ensure_hopper_shadow_state(
        self,
        shadow_key: DeepseekV4AttentionType,
        num_blocks: int,
        tokens_per_block: int,
        bytes_per_token: int,
        device: torch.device,
    ) -> None:
        if not hasattr(self, "_fp8_shadows"):
            self._fp8_shadows = {}
            self._fp8_block_fill_gpu = {}
            self._fp8_update_grids = {}
            self._fp8_offsets_cache = {}
            self._fp8_scale_pad_buf = torch.zeros(1, 8, dtype=torch.uint8, device=device)

        required_rows = num_blocks + 1
        shadow = self._fp8_shadows.get(shadow_key)
        if shadow is None or shadow.shape[0] != required_rows:
            self._fp8_shadows[shadow_key] = torch.zeros(
                required_rows,
                tokens_per_block * bytes_per_token,
                dtype=torch.uint8,
                device=device,
            )

        block_fill = self._fp8_block_fill_gpu.get(shadow_key)
        if block_fill is None or block_fill.shape[0] != num_blocks:
            self._fp8_block_fill_gpu[shadow_key] = torch.zeros(
                num_blocks, dtype=torch.int32, device=device
            )

    def _get_fp8_shadow_update_grid(
        self,
        num_requests: int,
        max_blocks: int,
        tokens_per_block: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (num_requests, max_blocks, tokens_per_block)
        cached = self._fp8_update_grids.get(key)
        if cached is not None:
            return cached
        total_slots = num_requests * max_blocks * tokens_per_block
        all_slots = torch.arange(total_slots, device=device, dtype=torch.long)
        token_grid = all_slots % tokens_per_block
        block_grid = (all_slots // tokens_per_block) % max_blocks
        request_grid = all_slots // (max_blocks * tokens_per_block)
        grids = (request_grid, block_grid, token_grid)
        self._fp8_update_grids[key] = grids
        return grids

    def _get_fp8_shadow_offsets(self, size: int, device: torch.device) -> torch.Tensor:
        cached = self._fp8_offsets_cache.get(size)
        if cached is None:
            cached = torch.arange(size, device=device, dtype=torch.long)
            self._fp8_offsets_cache[size] = cached
        return cached

    def _get_fp8_scale_padding(self, num_slots: int, device: torch.device) -> torch.Tensor:
        if self._fp8_scale_pad_buf.shape[0] < num_slots:
            self._fp8_scale_pad_buf = torch.zeros(num_slots, 8, dtype=torch.uint8, device=device)
        return self._fp8_scale_pad_buf[:num_slots]
