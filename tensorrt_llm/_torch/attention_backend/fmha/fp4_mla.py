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

from dataclasses import replace
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.fp4_mla import (
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
)
from tensorrt_llm._torch.attention_backend.fp4_mla.fp4_mla_context import (
    _FP8_CONTEXT_ATTN_ATTR,
    _FP8_CONTEXT_SCRATCH_ATTR,
    _build_fp8_mla_context_attn,
    _execute_fp8_context_with_cache_update,
    _Fp8MlaContextScratch,
    _get_fp8_mla_context_metadata,
    require_fp4_mla_fp8_context_support,
)
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm.bindings import DataType

from .fallback import FallbackFmha
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class Fp4MlaFmha(PhasedFmha):
    """TRTLLM FMHA library for FP4 MLA context and no-dequant decode."""

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        return attn.is_mla_enable and attn.has_fp4_kv_cache

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        self._validate_request(k, v, metadata, forward_args)
        super().forward(q, k, v, metadata, forward_args)

    def _validate_request(
        self,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        if forward_args.output_sf is not None:
            raise NotImplementedError("FP4 MLA does not support quantized attention output.")
        if forward_args.attention_mask != PredefinedAttentionMask.CAUSAL:
            raise NotImplementedError("FP4 MLA requires a causal attention mask.")
        if forward_args.attention_mask_data is not None:
            raise NotImplementedError("FP4 MLA does not support custom attention masks.")
        if forward_args.attention_sinks is not None:
            raise NotImplementedError("FP4 MLA does not support attention sinks.")

        sparse_prediction = forward_args.sparse_prediction
        sparse_params = self.attn.sparse_params
        uses_spcompress = getattr(sparse_params, "uses_spcompress", None)
        if (
            (
                sparse_prediction.sparse_kv_indices is not None
                and sparse_prediction.sparse_kv_indices.numel() > 0
            )
            or (
                sparse_prediction.sparse_attn_indices is not None
                and sparse_prediction.sparse_attn_indices.numel() > 0
            )
            or metadata.num_sparse_topk > 0
            or uses_spcompress
        ):
            raise NotImplementedError("FP4 MLA does not support sparse attention.")

        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None:
            raise RuntimeError("FP4 MLA requires a KV cache manager.")
        if kv_cache_manager.dtype != DataType.NVFP4:
            raise RuntimeError("FP4 MLA requires NVFP4 KV cache storage.")
        if kv_cache_manager.kv_factor != 1:
            raise RuntimeError("FP4 MLA requires a SELF-K-only KV cache.")
        if metadata.high_precision_kv_pool is None:
            raise RuntimeError("FP4 MLA requires the high-precision KV pool.")
        if metadata.fp4_mla_v_scale_pool is None:
            raise RuntimeError("FP4 MLA requires the V-scale pool.")
        if metadata.beam_width != 1:
            raise NotImplementedError("FP4 MLA does not support beam search.")

        attention_input_type = forward_args.attention_input_type
        if attention_input_type == AttentionInputType.context_only:
            if k is None or v is None:
                raise RuntimeError("FP4 MLA context requires expanded K and V tensors.")
            return
        if attention_input_type == AttentionInputType.generation_only:
            if k is not None or v is not None:
                raise RuntimeError("FP4 MLA generation expects a fused query input.")
            return
        raise NotImplementedError("FP4 MLA requires a context-only or generation-only call.")

    def run_mla_context(self, params: FmhaParams) -> None:
        attn = params.attn
        metadata = params.meta
        forward_args = params.fwd
        q = params.qkv_input
        k = params.key_input
        v = params.value_input
        output = params.context_buf
        if q is None or k is None or v is None:
            raise RuntimeError("FP4 MLA context requires expanded Q, K, and V tensors.")
        if output is None:
            raise RuntimeError("FP4 MLA context requires an output buffer.")
        if forward_args.latent_cache is None:
            raise RuntimeError("FP4 MLA context requires latent_cache.")
        if metadata.positions is None:
            raise RuntimeError("FP4 MLA context requires token positions.")
        if metadata.num_contexts <= 0:
            raise RuntimeError("FP4 MLA context requires context requests.")
        if getattr(metadata, "num_ctx_cached_tokens", 0) != 0:
            raise NotImplementedError("FP4 MLA does not support cached-context prefill.")
        if q.shape[0] != metadata.num_ctx_tokens:
            raise RuntimeError("FP4 MLA context query token count must match num_ctx_tokens.")
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            raise RuntimeError("FP4 MLA context Q/K/V token counts do not match.")

        require_fp4_mla_fp8_context_support()
        if metadata.is_cuda_graph:
            raise NotImplementedError(
                "FP4 MLA context does not support CUDA graphs with TRT-LLM FP8 FMHA."
            )

        num_tokens = q.shape[0]
        output = output.view(num_tokens, -1)
        local_layer = attn.get_local_layer_idx(metadata)
        kv_lora_rank = attn.kv_lora_rank or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0

        attn._ensure_rope_table_size(metadata.max_seq_len)
        latent_cache = forward_args.latent_cache[:num_tokens]

        def update_fp4_cache() -> None:
            hp_pool_updated = scatter_fp4_mla_kv_cache(
                metadata,
                latent_cache,
                attn.layer_idx,
                token_offset=0,
                phase="context",
                local_layer=local_layer,
                v_head_dim=kv_lora_rank,
                rotary_cos_sin=attn.rotary_cos_sin,
            )
            if not hp_pool_updated:
                raise RuntimeError("Fused FP4 MLA context scatter did not update the HP pool.")

        kv_cache_manager = metadata.kv_cache_manager
        if kv_cache_manager is None:
            raise RuntimeError("FP8 MLA context scratch requires a KV cache manager.")
        scratch = getattr(kv_cache_manager, _FP8_CONTEXT_SCRATCH_ATTR, None)
        scratch_head_dim = kv_lora_rank + qk_rope_head_dim
        if not isinstance(scratch, _Fp8MlaContextScratch) or not scratch.matches(
            metadata,
            device=q.device,
            head_dim=scratch_head_dim,
        ):
            scratch = _Fp8MlaContextScratch.create(
                metadata,
                device=q.device,
                head_dim=scratch_head_dim,
            )
            setattr(kv_cache_manager, _FP8_CONTEXT_SCRATCH_ATTR, scratch)

        fp8_attention = getattr(attn, _FP8_CONTEXT_ATTN_ATTR, None)
        if fp8_attention is None:
            fp8_attention = _build_fp8_mla_context_attn(attn)
            fp8_attention.fmha_libs = [FallbackFmha(fp8_attention)]
            setattr(attn, _FP8_CONTEXT_ATTN_ATTR, fp8_attention)
        fp8_attention.rotary_inv_freq = attn.rotary_inv_freq
        fp8_attention.rotary_cos_sin = attn.rotary_cos_sin
        fp8_metadata = _get_fp8_mla_context_metadata(metadata, scratch)
        fp8_forward_args = replace(
            forward_args,
            output=output,
            output_sf=None,
            kv_scale_orig_quant=None,
            kv_scale_quant_orig=None,
        )

        def run_fp8_context() -> None:
            fp8_attention.forward(q, k, v, fp8_metadata, fp8_forward_args)

        _execute_fp8_context_with_cache_update(
            run_fp8_context,
            update_fp4_cache,
            scratch.cache_stream,
            scratch.cache_start_event,
            scratch.cache_done_event,
        )

    def run_mla_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        metadata = params.meta
        forward_args = params.fwd
        q = params.qkv_input
        output = params.context_buf
        if q is None:
            raise RuntimeError("FP4 MLA generation requires a fused query input.")
        if output is None:
            raise RuntimeError("FP4 MLA generation requires an output buffer.")
        if forward_args.latent_cache is None:
            raise RuntimeError("FP4 MLA generation requires latent_cache.")
        if metadata.num_generations <= 0:
            raise RuntimeError("FP4 MLA generation requires generation requests.")

        local_layer = attn.get_local_layer_idx(metadata)
        kv_lora_rank = attn.kv_lora_rank or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        fused_head_dim = kv_lora_rank + qk_rope_head_dim

        if not bool(getattr(metadata, "_fp4_mla_generation_cache_scattered", False)):
            raise RuntimeError(
                "FP4 MLA generation requires fused RoPE/Q quantization and "
                "cache/HP-pool update before attention."
            )
        metadata._fp4_mla_generation_cache_scattered = False
        query = q.view(q.shape[0], attn.num_heads, fused_head_dim)
        output_view = output.view(q.shape[0], attn.num_heads, kv_lora_rank)
        sm_scale = 1.0 / (attn.q_scaling * ((attn.qk_nope_head_dim or 0) + qk_rope_head_dim) ** 0.5)
        prequantized_q = getattr(metadata, "_fp4_mla_prequantized_q", None)
        prequantized_q_sf = getattr(metadata, "_fp4_mla_prequantized_q_sf", None)
        q_batch_capacity = getattr(metadata, "_fp4_mla_q_batch_capacity", None)
        try:
            run_fp4_mla_attention_decode(
                metadata,
                attn.layer_idx,
                local_layer,
                query,
                output_view,
                sm_scale=sm_scale,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                prequantized_q=prequantized_q,
                prequantized_q_sf=prequantized_q_sf,
                q_batch_capacity=q_batch_capacity,
            )
        finally:
            metadata._fp4_mla_prequantized_q = None
            metadata._fp4_mla_prequantized_q_sf = None
            metadata._fp4_mla_q_batch_capacity = None


__all__ = ["Fp4MlaFmha"]
