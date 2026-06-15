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

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.fp4_mla import (
    FP4_MLA_Q_RESIDUAL_DIM,
    FP4_MLA_TOKENS_PER_BLOCK,
    HP_BLOCK_SIZE,
    apply_fp4_mla_rope,
    run_fp4_mla_attention_decode,
    scatter_fp4_mla_kv_cache,
    update_hp_kv_for_fp4_mla,
)
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.bindings import DataType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class Fp4MlaFmha(PhasedFmha):
    """TRTLLM FMHA library for the no-dequant NVFP4 MLA decode kernel."""

    SUPPORTED_Q_DTYPES = {torch.bfloat16, torch.float8_e4m3fn}
    SUPPORTED_CONTEXT_DTYPES = {torch.float16, torch.bfloat16}
    SUPPORTED_OUTPUT_DTYPES = {torch.float16, torch.bfloat16}

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        if not attn.is_mla_enable:
            logger.debug("FP4 MLA FMHA is unavailable: requires MLA.")
            return False
        if not QuantMode(attn.quant_mode).has_fp4_kv_cache():
            logger.debug("FP4 MLA FMHA is unavailable: requires NVFP4 KV cache quantization.")
            return False
        if attn.attention_chunk_size not in (None, 0):
            logger.debug("FP4 MLA FMHA is unavailable: chunked attention is not supported.")
            return False
        if attn.predicted_tokens_per_seq > HP_BLOCK_SIZE:
            logger.debug(
                "FP4 MLA FMHA is unavailable: linear MTP length exceeds "
                "FP4 MLA HP rollback support."
            )
            return False
        if attn.kv_lora_rank is None or attn.qk_rope_head_dim is None:
            logger.debug("FP4 MLA FMHA is unavailable: missing MLA dimensions.")
            return False
        if attn.qk_nope_head_dim is None or attn.v_head_dim is None:
            logger.debug("FP4 MLA FMHA is unavailable: missing MLA context dimensions.")
            return False
        if attn.qk_rope_head_dim != FP4_MLA_Q_RESIDUAL_DIM:
            logger.debug(
                f"FP4 MLA FMHA is unavailable: requires qk_rope_head_dim="
                f"{FP4_MLA_Q_RESIDUAL_DIM}, got {attn.qk_rope_head_dim}."
            )
            return False
        context_head_dim = attn.qk_nope_head_dim + attn.qk_rope_head_dim
        fused_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        if attn.head_dim not in (context_head_dim, fused_head_dim):
            logger.debug(
                "FP4 MLA FMHA is unavailable: head_dim must equal either "
                f"qk_nope_head_dim + qk_rope_head_dim ({context_head_dim}) or "
                f"kv_lora_rank + qk_rope_head_dim ({fused_head_dim})."
            )
            return False
        sm = get_sm_version()
        if not is_sm_100f(sm):
            logger.debug(f"FP4 MLA FMHA is unavailable: requires SM100 or SM103, got SM{sm}.")
            return False
        if not hasattr(torch.ops, "trtllm") or not hasattr(
            torch.ops.trtllm, "fp4_quantize_with_residual"
        ):
            logger.debug("FP4 MLA FMHA is unavailable: missing trtllm FP4 quantization op.")
            return False
        return True

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q, k, v, self.attn, metadata, forward_args
        )
        if not supported:
            logger.debug(f"FP4 MLA FMHA does not support request: {reason}")
        return supported

    @classmethod
    def _is_supported_with_reason(
        cls,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Tuple[bool, str]:
        if fwd.attention_input_type == AttentionInputType.context_only:
            return cls._is_context_supported_with_reason(q, k, v, attn, meta, fwd)

        if fwd.attention_input_type != AttentionInputType.generation_only:
            return False, "supports generation-only attention."
        if meta.num_generations <= 0:
            return False, "requires generation requests."
        if k is not None or v is not None:
            return False, "expects fused MLA query input."
        if fwd.output is None:
            return False, "requires output."
        if fwd.latent_cache is None:
            return False, "requires latent_cache."
        if q.dtype not in cls.SUPPORTED_Q_DTYPES:
            return False, f"unsupported query dtype {q.dtype}."
        if fwd.output.dtype not in cls.SUPPORTED_OUTPUT_DTYPES:
            return False, f"unsupported output dtype {fwd.output.dtype}."
        if fwd.output_sf is not None:
            return False, "does not support quantized attention output."
        if fwd.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False, "requires causal mask."
        if fwd.attention_mask_data is not None:
            return False, "does not support custom attention masks."
        if fwd.attention_sinks is not None:
            return False, "does not support attention sinks."
        if fwd.sage_attn_num_elts_per_blk_q > 0 or fwd.sage_attn_num_elts_per_blk_k > 0:
            return False, "does not support sage attention."
        if fwd.sage_attn_num_elts_per_blk_v > 0:
            return False, "does not support sage attention."
        sparse = fwd.sparse_prediction
        if (
            (sparse.sparse_kv_indices is not None and sparse.sparse_kv_indices.numel() > 0)
            or (sparse.sparse_attn_indices is not None and sparse.sparse_attn_indices.numel() > 0)
            or meta.num_sparse_topk > 0
        ):
            return False, "does not support sparse attention."
        if meta.helix_position_offsets is not None:
            return False, "does not support helix parallelism."
        if meta.use_spec_decoding and meta.is_spec_dec_tree:
            return False, "does not support speculative decoding trees."
        if meta.kv_cache_manager is None:
            return False, "requires a KV cache manager."
        if meta.kv_cache_manager.dtype != DataType.NVFP4:
            return False, f"requires NVFP4 KV cache storage, got {meta.kv_cache_manager.dtype}."
        if meta.kv_cache_manager.kv_factor != 1:
            return False, "requires MLA SELF-K-only KV cache."
        if meta.kv_cache_block_offsets is None:
            return False, "requires paged KV cache block offsets."
        if meta.high_precision_kv_pool is None:
            return False, "requires high-precision KV pool."
        if meta.fp4_mla_v_scale_pool is None:
            return False, "requires FP4 MLA V-scale pool."
        if meta.batch_indices is None or meta.positions is None:
            return False, "requires FP4 MLA append metadata."
        if (
            meta._paged_kv_indptr is None
            or meta.paged_kv_indptr_decode is None
            or meta._paged_kv_indices is None
        ):
            return False, "requires FP4 MLA page metadata."
        if meta.tokens_per_block != FP4_MLA_TOKENS_PER_BLOCK:
            return (
                False,
                f"requires tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK}, "
                f"got {meta.tokens_per_block}.",
            )
        if fwd.attention_window_size is not None and fwd.attention_window_size < meta.max_seq_len:
            return False, "does not support sliding-window attention."
        if meta.beam_width != 1:
            return False, f"does not support beam search, got beam_width={meta.beam_width}."
        fused_head_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        if q.shape[-1] != attn.num_heads * fused_head_dim:
            return False, f"unexpected fused query hidden size {q.shape[-1]}."
        if fwd.latent_cache.shape[-1] != fused_head_dim:
            return False, f"unexpected latent_cache hidden size {fwd.latent_cache.shape[-1]}."
        if q.shape[0] != fwd.latent_cache.shape[0]:
            return False, "query and latent_cache token counts do not match."
        if q.shape[0] < meta.num_generations:
            return False, "not enough query tokens for generation batch."
        if q.shape[0] % meta.num_generations != 0:
            return False, "requires uniform linear MTP generation length."

        return True, ""

    @classmethod
    def _is_context_supported_with_reason(
        cls,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Tuple[bool, str]:
        if meta.num_contexts <= 0:
            return False, "requires context requests."
        if getattr(meta, "num_ctx_cached_tokens", 0) != 0:
            return False, "does not support cached-context FP4 MLA prefill."
        if k is None or v is None:
            return False, "requires expanded context K and V tensors."
        if fwd.output is None:
            return False, "requires output."
        if fwd.latent_cache is None:
            return False, "requires latent_cache."
        if q.dtype not in cls.SUPPORTED_CONTEXT_DTYPES:
            return False, f"unsupported context query dtype {q.dtype}."
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return False, "requires matching context q/k/v dtypes."
        if fwd.output.dtype not in cls.SUPPORTED_OUTPUT_DTYPES:
            return False, f"unsupported output dtype {fwd.output.dtype}."
        if fwd.output_sf is not None:
            return False, "does not support quantized context output."
        if fwd.attention_mask != PredefinedAttentionMask.CAUSAL:
            return False, "requires causal mask."
        if fwd.attention_mask_data is not None:
            return False, "does not support custom attention masks."
        if fwd.attention_sinks is not None:
            return False, "does not support attention sinks."
        if fwd.sage_attn_num_elts_per_blk_q > 0 or fwd.sage_attn_num_elts_per_blk_k > 0:
            return False, "does not support sage attention."
        if fwd.sage_attn_num_elts_per_blk_v > 0:
            return False, "does not support sage attention."
        sparse = fwd.sparse_prediction
        if (
            (sparse.sparse_kv_indices is not None and sparse.sparse_kv_indices.numel() > 0)
            or (sparse.sparse_attn_indices is not None and sparse.sparse_attn_indices.numel() > 0)
            or meta.num_sparse_topk > 0
        ):
            return False, "does not support sparse attention."
        if meta.helix_position_offsets is not None:
            return False, "does not support helix parallelism."
        if meta.kv_cache_manager is None:
            return False, "requires a KV cache manager."
        if meta.kv_cache_manager.dtype != DataType.NVFP4:
            return False, f"requires NVFP4 KV cache storage, got {meta.kv_cache_manager.dtype}."
        if meta.kv_cache_manager.kv_factor != 1:
            return False, "requires MLA SELF-K-only KV cache."
        if meta.kv_cache_block_offsets is None:
            return False, "requires paged KV cache block offsets."
        if meta.high_precision_kv_pool is None:
            return False, "requires high-precision KV pool."
        if meta.fp4_mla_v_scale_pool is None:
            return False, "requires FP4 MLA V-scale pool."
        if meta.batch_indices is None or meta.positions is None:
            return False, "requires FP4 MLA append metadata."
        if meta.paged_kv_indptr_decode is None or meta._paged_kv_indices is None:
            return False, "requires FP4 MLA page metadata."
        if meta.tokens_per_block != FP4_MLA_TOKENS_PER_BLOCK:
            return (
                False,
                f"requires tokens_per_block={FP4_MLA_TOKENS_PER_BLOCK}, "
                f"got {meta.tokens_per_block}.",
            )
        if fwd.attention_window_size is not None and fwd.attention_window_size < meta.max_seq_len:
            return False, "does not support sliding-window attention."
        if meta.beam_width != 1:
            return False, f"does not support beam search, got beam_width={meta.beam_width}."
        qk_head_dim = attn.qk_nope_head_dim + attn.qk_rope_head_dim
        if q.shape[-1] != attn.num_heads * qk_head_dim:
            return False, f"unexpected context query hidden size {q.shape[-1]}."
        if k.shape[-1] != attn.num_heads * qk_head_dim:
            return False, f"unexpected context key hidden size {k.shape[-1]}."
        if v.shape[-1] != attn.num_heads * attn.v_head_dim:
            return False, f"unexpected context value hidden size {v.shape[-1]}."
        if fwd.latent_cache.shape[-1] != attn.kv_lora_rank + attn.qk_rope_head_dim:
            return False, f"unexpected latent_cache hidden size {fwd.latent_cache.shape[-1]}."
        if q.shape[0] != meta.num_ctx_tokens:
            return False, "query token count must match num_ctx_tokens."
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            return False, "context q/k/v token counts do not match."
        if fwd.latent_cache.shape[0] < q.shape[0]:
            return False, "latent_cache does not contain all context tokens."

        return True, ""

    def run_mla_context(self, params: FmhaParams) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if params.qkv_input is None:
            raise RuntimeError("FP4 MLA context requires q input.")
        if params.k_input is None or params.v_input is None:
            raise RuntimeError("FP4 MLA context requires expanded k/v inputs.")
        if params.context_buf is None:
            raise RuntimeError("FP4 MLA context requires context_buf.")
        if fwd.latent_cache is None:
            raise RuntimeError("FP4 MLA context requires latent_cache.")
        if meta.positions is None:
            raise RuntimeError("FP4 MLA context requires positions.")

        local_layer = attn.get_local_layer_idx(meta)
        kv_lora_rank = attn.kv_lora_rank or 0
        qk_nope_head_dim = attn.qk_nope_head_dim or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        v_head_dim = attn.v_head_dim or 0
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        num_tokens = params.num_tokens

        positions = meta.positions[:num_tokens]
        q_ctx = params.qkv_input.view(num_tokens, attn.num_heads, qk_head_dim)
        k_ctx = params.k_input.view(num_tokens, attn.num_heads, qk_head_dim)
        v_ctx = params.v_input.view(num_tokens, attn.num_heads, v_head_dim)

        q_nope = q_ctx[..., :qk_nope_head_dim]
        q_pe = q_ctx[..., qk_nope_head_dim:]
        k_nope = k_ctx[..., :qk_nope_head_dim]
        k_pe = fwd.latent_cache[:num_tokens, kv_lora_rank:].unsqueeze(1)

        q_pe = apply_fp4_mla_rope(
            q_pe,
            positions,
            attn.rotary_cos_sin,
            attn.rope_params.max_positions,
            qk_rope_head_dim,
        )
        k_pe = apply_fp4_mla_rope(
            k_pe,
            positions,
            attn.rotary_cos_sin,
            attn.rope_params.max_positions,
            qk_rope_head_dim,
        ).squeeze(1)

        latent_cache = torch.empty_like(fwd.latent_cache[:num_tokens])
        latent_cache[..., :kv_lora_rank].copy_(fwd.latent_cache[:num_tokens, :kv_lora_rank])
        latent_cache[..., kv_lora_rank:].copy_(k_pe)
        scatter_fp4_mla_kv_cache(
            meta,
            latent_cache,
            attn.layer_idx,
            token_offset=0,
            phase="context",
            local_layer=local_layer,
            v_head_dim=kv_lora_rank,
        )
        update_hp_kv_for_fp4_mla(meta, latent_cache, local_layer, phase="context")

        q_ctx = torch.cat((q_nope, q_pe), dim=-1)
        k_ctx = torch.cat((k_nope, k_pe.unsqueeze(1).expand(-1, attn.num_heads, -1)), dim=-1)
        output = params.context_buf.view(num_tokens, attn.num_heads, v_head_dim)
        sm_scale = 1.0 / (attn.q_scaling * qk_head_dim**0.5)

        host_context_lengths = meta.prompt_lens_cpu_runtime[: meta.num_contexts].tolist()
        token_offset = 0
        for context_len in host_context_lengths:
            if context_len == 0:
                continue
            next_offset = token_offset + int(context_len)
            q_seq = q_ctx[token_offset:next_offset].transpose(0, 1).unsqueeze(0)
            k_seq = k_ctx[token_offset:next_offset].transpose(0, 1).unsqueeze(0)
            v_seq = v_ctx[token_offset:next_offset].transpose(0, 1).unsqueeze(0)
            out_seq = F.scaled_dot_product_attention(
                q_seq,
                k_seq,
                v_seq,
                is_causal=True,
                scale=sm_scale,
            )
            output[token_offset:next_offset].copy_(out_seq.squeeze(0).transpose(0, 1))
            token_offset = next_offset

    def run_mla_generation(self, params: FmhaParams) -> None:
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        if params.qkv_input is None:
            raise RuntimeError("FP4 MLA generation requires qkv_input.")
        if params.context_buf is None:
            raise RuntimeError("FP4 MLA generation requires context_buf.")
        if fwd.latent_cache is None:
            raise RuntimeError("FP4 MLA generation requires latent_cache.")

        local_layer = attn.get_local_layer_idx(meta)
        kv_lora_rank = attn.kv_lora_rank or 0
        qk_rope_head_dim = attn.qk_rope_head_dim or 0
        fused_head_dim = kv_lora_rank + qk_rope_head_dim

        scatter_fp4_mla_kv_cache(
            meta,
            fwd.latent_cache,
            attn.layer_idx,
            token_offset=getattr(meta, "num_ctx_tokens", 0),
            phase="generation",
            local_layer=local_layer,
            v_head_dim=kv_lora_rank,
        )
        update_hp_kv_for_fp4_mla(meta, fwd.latent_cache, local_layer, phase="generation")

        query = params.qkv_input.view(params.num_tokens, attn.num_heads, fused_head_dim)
        q_nope = query[..., :kv_lora_rank]
        q_pe = query[..., kv_lora_rank:]
        output = params.context_buf.view(params.num_tokens, attn.num_heads, kv_lora_rank)
        sm_scale = 1.0 / (attn.q_scaling * (attn.qk_nope_head_dim + qk_rope_head_dim) ** 0.5)
        run_fp4_mla_attention_decode(
            meta,
            attn.layer_idx,
            local_layer,
            q_nope,
            q_pe,
            output,
            sm_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
