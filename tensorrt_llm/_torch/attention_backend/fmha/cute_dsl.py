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
"""CuTe DSL MLA decode FMHA library."""

import math
import os
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )

_LOG2_E = math.log2(math.e)


class CuteDslMlaFmha(PhasedFmha):
    """Blackwell CuTe DSL FMHA library for decode-only MLA."""

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        if not IS_CUTLASS_DSL_AVAILABLE:
            logger.debug("CuTe DSL MLA FMHA is unavailable: nvidia-cutlass-dsl is not installed.")
            return False

        sm = get_sm_version()
        if sm not in (100, 103):
            logger.debug(f"CuTe DSL MLA FMHA is unavailable: requires SM100 or SM103, got SM{sm}.")
            return False

        if not attn.is_mla_enable:
            logger.debug("CuTe DSL MLA FMHA is unavailable: only MLA is supported.")
            return False
        if attn.predicted_tokens_per_seq is None or not (1 <= attn.predicted_tokens_per_seq <= 4):
            logger.debug(
                "CuTe DSL MLA FMHA is unavailable: predicted_tokens_per_seq "
                f"must be in [1, 4], got {attn.predicted_tokens_per_seq}."
            )
            return False
        if attn.kv_lora_rank is None or attn.kv_lora_rank <= 0:
            logger.debug("CuTe DSL MLA FMHA is unavailable: kv_lora_rank must be positive.")
            return False
        if attn.qk_rope_head_dim is None or attn.qk_rope_head_dim <= 0:
            logger.debug("CuTe DSL MLA FMHA is unavailable: qk_rope_head_dim must be positive.")
            return False
        if attn.qk_nope_head_dim is None or attn.qk_nope_head_dim <= 0:
            logger.debug("CuTe DSL MLA FMHA is unavailable: qk_nope_head_dim must be positive.")
            return False
        if attn.kv_lora_rank != 512 or attn.qk_rope_head_dim != 64:
            logger.debug(
                "CuTe DSL MLA FMHA is unavailable: kernels require kv_lora_rank=512 and "
                f"qk_rope_head_dim=64, got kv_lora_rank={attn.kv_lora_rank}, "
                f"qk_rope_head_dim={attn.qk_rope_head_dim}."
            )
            return False
        if attn.num_heads > 128:
            logger.debug(
                f"CuTe DSL MLA FMHA is unavailable: num_heads must be <= 128, got {attn.num_heads}."
            )
            return False

        return True

    @staticmethod
    def _get_kernel_dtype(attn: "TrtllmAttention", q: torch.Tensor) -> Optional[torch.dtype]:
        if getattr(attn, "has_fp8_kv_cache", False):
            return torch.float8_e4m3fn
        if q.dtype in (torch.float16, torch.bfloat16):
            return q.dtype
        return None

    @staticmethod
    def _select_page_table_layer(
        block_offsets: torch.Tensor,
        layer_idx: int,
        host_kv_cache_pool_mapping: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if block_offsets.dim() == 4:
            if block_offsets.shape[2] < 1:
                return None
            if host_kv_cache_pool_mapping is not None:
                if layer_idx >= host_kv_cache_pool_mapping.shape[0]:
                    return None
                pool_idx = int(host_kv_cache_pool_mapping[layer_idx, 0])
            else:
                pool_idx = layer_idx if block_offsets.shape[0] > 1 else 0
            if pool_idx >= block_offsets.shape[0]:
                return None
            return block_offsets[pool_idx, :, 0, :]
        if block_offsets.dim() == 3:
            if block_offsets.shape[1] < 1:
                return None
            return block_offsets[:, 0, :]
        if block_offsets.dim() == 2:
            return block_offsets
        return None

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> bool:
        supported, reason = self._is_supported_with_reason(
            q,
            self.attn,
            metadata,
            forward_args,
        )
        if not supported:
            logger.debug(f"CuTe DSL MLA FMHA does not support request: {reason}")
        return supported

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> tuple[bool, str]:
        if fwd.attention_input_type != AttentionInputType.generation_only:
            return False, "CuTe DSL MLA FMHA only supports generation-only attention."
        if meta.num_contexts != 0 or meta.num_generations <= 0:
            return False, "CuTe DSL MLA FMHA only supports decode-only batches."
        if meta.beam_width != 1:
            return False, f"Beam search is not supported, got beam_width={meta.beam_width}."
        if (
            fwd.attention_mask == CustomAttentionMask.CUSTOM
            or fwd.attention_mask_data is not None
            or getattr(meta, "use_spec_decoding", False)
            or getattr(meta, "is_spec_decoding_enabled", False)
        ):
            return False, "CuTe DSL MLA FMHA does not support custom/speculative masks."
        if q.shape[0] % meta.num_generations != 0:
            return (
                False,
                f"num_tokens ({q.shape[0]}) must be divisible by "
                f"num_generations ({meta.num_generations}).",
            )
        seq_len_q = q.shape[0] // meta.num_generations
        if not (1 <= seq_len_q <= 4):
            return False, f"Only query lengths in [1, 4] are supported, got {seq_len_q}."
        if meta.kv_cache_block_offsets is None:
            return False, "Paged KV block offsets are required."
        page_table_layer = self._select_page_table_layer(
            meta.kv_cache_block_offsets,
            attn.layer_idx,
            meta.host_kv_cache_pool_mapping,
        )
        if page_table_layer is None:
            return (
                False,
                "Unsupported KV block offsets shape "
                f"{tuple(meta.kv_cache_block_offsets.shape)} for layer_idx={attn.layer_idx}.",
            )
        if meta.kv_cache_manager is None:
            return False, "KV cache manager is required."
        if fwd.latent_cache is None:
            return False, "latent_cache is required."
        if fwd.output is None:
            return False, "output is required."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block is None:
            tokens_per_block = getattr(meta.kv_cache_manager, "tokens_per_block", 0)
        if tokens_per_block <= 1 or 128 % tokens_per_block != 0:
            return (
                False,
                f"tokens_per_block must divide 128 and be greater than 1, got {tokens_per_block}.",
            )

        kernel_dtype = self._get_kernel_dtype(attn, q)
        if kernel_dtype is None:
            return (
                False,
                f"Unsupported dtype combination: q={q.dtype}, "
                f"has_fp8_kv_cache={getattr(attn, 'has_fp8_kv_cache', False)}.",
            )
        if kernel_dtype == torch.float8_e4m3fn and (
            fwd.quant_q_buffer is None or fwd.mla_bmm1_scale is None or fwd.mla_bmm2_scale is None
        ):
            return (
                False,
                "FP8 CuTe DSL MLA decode requires quant_q_buffer, "
                "mla_bmm1_scale, and mla_bmm2_scale from MLA RoPE generation.",
            )
        if kernel_dtype in (torch.float16, torch.bfloat16):
            kv_pool_dtype = meta.kv_cache_manager.get_buffers(attn.layer_idx).dtype
            if kv_pool_dtype != kernel_dtype:
                return (
                    False,
                    f"CuTe DSL MLA {kernel_dtype} fast path requires matching "
                    f"KV cache dtype, got {kv_pool_dtype}.",
                )

        return True, ""

    def _run_mla_decode(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        params: FmhaParams,
        kernel_dtype: torch.dtype,
    ) -> None:
        attn = params.attn
        meta = params.meta

        if kernel_dtype == torch.float8_e4m3fn:
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell
        elif kernel_dtype in (torch.float16, torch.bfloat16):
            op = torch.ops.trtllm.cute_dsl_mla_decode_fp16_blackwell
        else:
            raise ValueError(
                f"CuTe DSL MLA FMHA got unsupported kernel_dtype={kernel_dtype}; "
                "expected torch.float8_e4m3fn, torch.float16, or torch.bfloat16."
            )

        num_tokens = q.shape[0]
        batch_size = params.num_requests
        seq_len_q = num_tokens // batch_size
        if seq_len_q * batch_size != num_tokens:
            raise RuntimeError(
                f"CuTe DSL MLA decode expects num_tokens ({num_tokens}) divisible by "
                f"batch_size ({batch_size})."
            )

        d_latent = attn.kv_lora_rank
        d_rope = attn.qk_rope_head_dim
        qk_nope_head_dim = attn.qk_nope_head_dim
        if d_latent is None or d_rope is None or qk_nope_head_dim is None:
            raise RuntimeError("CuTe DSL MLA decode requires complete MLA dimensions.")

        num_heads = attn.num_heads
        page_size = params.tokens_per_block

        if kernel_dtype == torch.float8_e4m3fn and params.fwd.quant_q_buffer is not None:
            q_kernel = params.fwd.quant_q_buffer.view(torch.float8_e4m3fn).view_as(q)
        else:
            q_kernel = q if q.dtype == kernel_dtype else q.to(kernel_dtype)
        q_view = q_kernel.view(batch_size, seq_len_q, num_heads, d_latent + d_rope)

        kv_pool = meta.kv_cache_manager.get_buffers(attn.layer_idx)
        if kernel_dtype in (torch.float16, torch.bfloat16) and kv_pool.dtype != kernel_dtype:
            raise RuntimeError(
                f"CuTe DSL MLA {kernel_dtype} fast path requires matching "
                f"KV cache dtype, got {kv_pool.dtype}."
            )
        # Paged-pool layout normalization for both KV cache managers.
        # KVCacheManagerV2 exposes each layer as a densely-packed page pool.
        # KVCacheManagerV1 exposes a per-layer view over one interleaved pool,
        # where dim-0 strides by all local layers. The CuTe DSL kernel addresses
        # pages with the packed block stride, so represent v1 as a packed
        # combined-slot view and fold this layer's slot offset into the page
        # table below.
        packed_block = 1
        for size in kv_pool.shape[1:]:
            packed_block *= size
        block_stride = kv_pool.stride(0)
        layers_in_pool = block_stride // packed_block if packed_block else 1
        layer_in_pool = 0
        if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
            layer_in_pool = kv_pool.storage_offset() // packed_block
            kv_pool = kv_pool.as_strided(
                (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
                (packed_block, *kv_pool.stride()[1:]),
                0,
            )
        else:
            layers_in_pool = 1

        kv_pool_typed = kv_pool.view(kernel_dtype)
        if kv_pool_typed.dim() != 5 or kv_pool_typed.shape[1] != 1 or kv_pool_typed.shape[3] != 1:
            raise RuntimeError(
                "CuTe DSL MLA decode expects KV cache layout "
                f"[num_pages, 1, page_size, 1, head_dim], got {tuple(kv_pool_typed.shape)}."
            )
        if kv_pool_typed.shape[2] != page_size or kv_pool_typed.shape[-1] < d_latent + d_rope:
            raise RuntimeError(
                "CuTe DSL MLA decode got incompatible KV cache shape "
                f"{tuple(kv_pool_typed.shape)} for page_size={page_size}, "
                f"kv_lora_rank={d_latent}, qk_rope_head_dim={d_rope}."
            )

        block_offsets = meta.kv_cache_block_offsets
        page_table_layer = self._select_page_table_layer(
            block_offsets,
            attn.layer_idx,
            meta.host_kv_cache_pool_mapping,
        )
        if page_table_layer is None:
            raise RuntimeError(
                "CuTe DSL MLA decode got unsupported KV block offsets shape "
                f"{tuple(block_offsets.shape)} for layer_idx={attn.layer_idx}."
            )
        cache_seqs_base = params.sequence_lengths.to(torch.int32)
        page_table = page_table_layer.transpose(0, 1).to(torch.int32)
        if layers_in_pool > 1:
            page_table = page_table + layer_in_pool

        if os.environ.get("TLLM_CUTE_DSL_DUMP"):
            print(
                "[CUTEDSL_DUMP] layer=%d kv_pool.shape=%s kv_pool.stride=%s "
                "contiguous=%s block_offsets.shape=%s page_table.shape=%s "
                "page_table=%s cache_seqs=%s"
                % (
                    attn.layer_idx,
                    tuple(kv_pool.shape),
                    tuple(kv_pool.stride()),
                    kv_pool.is_contiguous(),
                    tuple(block_offsets.shape),
                    tuple(page_table.shape),
                    page_table.t().tolist() if page_table.numel() < 64 else "(big)",
                    cache_seqs_base[:8].tolist(),
                ),
                flush=True,
            )

        # KVCacheManager exposes NHD pages as [num_pages, 1, page_size, 1, head_dim].
        # The CuTe DSL kernel consumes a paged [page_size, dim, num_pages] view
        # with the dim axis contiguous.
        kv_pages = kv_pool_typed[:, 0, :, 0, : d_latent + d_rope]
        c_pool_latent = kv_pages[..., :d_latent].permute(1, 2, 0)
        c_pool_rope = kv_pages[..., d_latent:].permute(1, 2, 0)

        block_split_kvs = torch.empty(0, dtype=torch.int32, device=q.device)
        split_kv = 1
        workspace = torch.empty(0, dtype=torch.float32, device=q.device)

        softmax_scale = float(1.0 / (math.sqrt(qk_nope_head_dim + d_rope) * attn.q_scaling))
        output_scale = 1.0
        if kernel_dtype == torch.float8_e4m3fn:
            if params.fwd.mla_bmm1_scale is None or params.fwd.mla_bmm2_scale is None:
                raise RuntimeError("FP8 CuTe DSL MLA decode requires MLA FP8 scales.")
            cached = getattr(self, "_cute_dsl_fp8_scale", None)
            if cached is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "CuTe DSL MLA FMHA: fp8 decode scale was not cached for "
                        f"layer {attn.layer_idx} before CUDA graph capture."
                    )
                softmax_scale = float(params.fwd.mla_bmm1_scale[1].item()) / _LOG2_E
                output_scale = float(params.fwd.mla_bmm2_scale[0].item())
                self._cute_dsl_fp8_scale = (softmax_scale, output_scale)
            else:
                softmax_scale, output_scale = cached
                if (
                    os.environ.get("TLLM_CUTE_DSL_SCALE_DUMP")
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    live_softmax_scale = float(params.fwd.mla_bmm1_scale[1].item()) / _LOG2_E
                    live_output_scale = float(params.fwd.mla_bmm2_scale[0].item())
                    print(
                        "[CUTEDSL_SCALE] layer=%d cached=(%.8f,%.8f) "
                        "live=(%.8f,%.8f) drift=%s"
                        % (
                            attn.layer_idx,
                            softmax_scale,
                            output_scale,
                            live_softmax_scale,
                            live_output_scale,
                            abs(live_softmax_scale - softmax_scale) > 1e-6
                            or abs(live_output_scale - output_scale) > 1e-6,
                        ),
                        flush=True,
                    )

        out_kernel_dtype = torch.bfloat16 if kernel_dtype == torch.float8_e4m3fn else kernel_dtype
        output_view = output.view(batch_size, seq_len_q, num_heads, d_latent)
        for query_idx in range(seq_len_q):
            q_step = q_view[:, query_idx : query_idx + 1, :, :]
            q_latent = q_step[..., :d_latent].permute(2, 3, 1, 0)
            q_rope = q_step[..., d_latent:].permute(2, 3, 1, 0)

            o_storage = torch.empty(
                (batch_size, 1, num_heads, d_latent),
                dtype=out_kernel_dtype,
                device=q.device,
            )
            o_kernel = o_storage.permute(2, 3, 1, 0)
            lse_storage = torch.empty(
                (batch_size, 1, num_heads),
                dtype=torch.float32,
                device=q.device,
            )
            lse = lse_storage.permute(2, 1, 0)

            # MLA RoPE generation has already appended all query tokens in this
            # step. For multi-query decode, trim the effective KV length so each
            # query attends only through its own generated token.
            cache_seqs = cache_seqs_base - (seq_len_q - query_idx - 1)

            op(
                q_latent,
                q_rope,
                c_pool_latent,
                c_pool_rope,
                page_table,
                cache_seqs,
                block_split_kvs,
                o_kernel,
                lse,
                workspace,
                num_heads,
                1,  # seq_len_q
                page_size,
                True,  # is_persistent
                True,  # is_var_seq
                False,  # is_var_split_kv
                split_kv,
                softmax_scale,
                output_scale,
            )

            attn_out = o_kernel.permute(3, 2, 0, 1).reshape(batch_size, num_heads, d_latent)
            output_view[:, query_idx, :, :].copy_(attn_out.to(output.dtype))

    def run_mla_generation(
        self,
        params: FmhaParams,
    ) -> None:
        if params.qkv_input is None:
            raise RuntimeError("CuTe DSL MLA generation requires qkv_input.")
        if params.context_buf is None:
            raise RuntimeError("CuTe DSL MLA generation requires context_buf.")
        if params.sequence_lengths is None:
            raise RuntimeError("CuTe DSL MLA generation requires sequence lengths.")

        kernel_dtype = self._get_kernel_dtype(params.attn, params.qkv_input)
        if kernel_dtype is None:
            raise RuntimeError("CuTe DSL MLA generation was selected for an unsupported dtype.")

        self._run_mla_decode(
            params.qkv_input,
            params.context_buf,
            params,
            kernel_dtype,
        )
