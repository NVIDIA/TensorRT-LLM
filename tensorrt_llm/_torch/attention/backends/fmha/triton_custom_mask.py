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

"""Triton custom-mask context attention for the TRT-LLM backend."""

import math
from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    CustomAttentionMask,
)
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha
from .utils import (
    get_attention_chunk_size,
    get_bmm1_scale,
    get_multi_processor_count_for_device,
    get_trtllm_gen_context_workspace_size,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class TritonCustomMaskFmha(PhasedFmha):
    """Run custom-mask context attention with Triton."""

    SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._multi_processor_count: Optional[int] = None

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        required_ops = (
            "get_trtllm_gen_context_workspace_layout",
            "trtllm_gen_context_preprocess",
        )
        missing_ops = [op for op in required_ops if not hasattr(thop, op)]
        if missing_ops:
            logger.debug(
                "Triton custom-mask FMHA is unavailable: missing fused "
                f"nanobind ops: {', '.join(missing_ops)}."
            )
            return False
        if attn.num_heads <= 0 or attn.num_kv_heads <= 0:
            return False
        if attn.num_heads % attn.num_kv_heads != 0:
            return False
        return True

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        supported, reason = self._check_support_with_reason(
            q,
            k,
            v,
            metadata,
            forward_args,
            phase=phase,
        )
        if not supported:
            logger.debug(f"Triton custom-mask FMHA does not support request: {reason}")
        return supported

    def _check_support_with_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> tuple[bool, str]:
        if phase != FmhaPhase.CONTEXT:
            return False, "Only context attention is supported."
        if forward_args.attention_mask != CustomAttentionMask.CUSTOM:
            return False, "Request does not use a custom attention mask."
        if forward_args.attention_mask_data is None:
            return False, "Custom attention requires attention_mask_data."
        if metadata.num_contexts <= 0:
            return False, "Custom attention requires at least one context request."
        if metadata.is_cross:
            return False, "Custom-mask cross attention is not supported."
        if metadata.kv_cache_block_offsets is None:
            return False, "Custom-mask TRT-LLM attention requires paged KV cache."
        if metadata.kv_layout != "HND":
            return False, "Custom-mask TRT-LLM attention requires HND KV-cache layout."
        if getattr(metadata.kv_cache_manager, "enable_swa_scratch_reuse", False):
            return False, "Custom-mask TRT-LLM attention does not support SWA scratch reuse."
        if self.attn.is_mla_enable:
            return False, "Custom-mask MLA is not supported."
        if not forward_args.is_fused_qkv or k is not None or v is not None:
            return False, "Custom-mask TRT-LLM attention requires fused QKV input."
        if q.dtype not in self.SUPPORTED_DTYPES:
            return False, f"Input dtype {q.dtype} is not supported."
        output = forward_args.output
        if output is None or output.dtype != q.dtype:
            return False, "Custom-mask Triton attention requires output to match the input dtype."
        if forward_args.output_sf is not None:
            return False, "Custom-mask Triton attention does not support NVFP4 output."
        if QuantMode(self.attn.quant_mode).has_kv_cache_quant():
            return False, "Custom-mask TRT-LLM attention does not support quantized KV cache."
        if forward_args.attention_sinks is not None:
            return False, "Custom-mask Triton attention does not support attention sinks."
        if (
            getattr(self.attn, "head_dim", 0) > 256
            and PositionEmbeddingType(self.attn.position_embedding_type).is_rope()
        ):
            return (
                False,
                "Custom-mask head dimensions above 256 require RoPE to be applied before the backend.",
            )

        return True, ""

    @classmethod
    def _get_multi_processor_count(cls, device: torch.device) -> int:
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("Triton custom-mask FMHA requires CUDA tensors.")
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        return get_multi_processor_count_for_device(device_index)

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        if self._multi_processor_count is None:
            self._multi_processor_count = self._get_multi_processor_count(q.device)

        attn = self.attn
        required_workspace_size = get_trtllm_gen_context_workspace_size(
            dtype=q.dtype,
            max_num_seq=metadata.max_num_requests,
            max_num_tokens=max(q.size(0), metadata.max_context_length),
            num_heads=attn.num_heads,
            head_size=attn.head_dim,
            rotary_embedding_dim=attn.rope_dim,
            fp8_context_fmha=False,
        )
        current_workspace_size = workspace.numel() * workspace.element_size()
        if current_workspace_size < required_workspace_size:
            if metadata.is_cuda_graph and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Attention CUDA graph workspace is smaller than the required size "
                    "for Triton custom-mask FMHA."
                )
            required_workspace_numel = math.ceil(required_workspace_size / workspace.element_size())
            workspace.resize_((required_workspace_numel,))

    def run_context(self, params: FmhaParams) -> None:
        """Preprocess fused QKV, then run the Triton custom-mask context kernel."""
        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        bmm1_scale = get_bmm1_scale(attn)
        attention_chunk_size = get_attention_chunk_size(attn)

        (
            q_processed,
            _,
            block_tables,
            _,
            _,
            _,
            _,
            cu_q_seqlens,
            _,
            _,
            _,
            window_left,
        ) = thop.trtllm_gen_context_preprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.context_lengths,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_inv_freq,
            attn.rotary_cos_sin,
            fwd.mrope_rotary_cos_sin,
            attn.local_layer_idx,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            int(AttentionMaskType.causal),
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            bmm1_scale,
            1.0,
            attention_chunk_size,
            False,  # fp8_context_fmha
            False,  # paged_context_fmha; keep processed QKV packed in-place
            False,  # is_mla_enable
            self._multi_processor_count,
            params.total_num_blocks,
            params.kv_factor,
            True,  # need_build_kv_cache_metadata
            None,  # cross_kv
            False,  # cross_attention
        )

        if block_tables is None:
            raise RuntimeError("Custom-mask TRT-LLM attention requires paged KV metadata.")
        if params.qkv_input is None or params.context_buf is None:
            raise RuntimeError(
                "Custom-mask TRT-LLM attention requires context QKV and output buffers."
            )
        if params.sequence_lengths is None or params.context_lengths is None:
            raise RuntimeError("Custom-mask TRT-LLM attention requires context sequence lengths.")

        q_size = attn.num_heads * attn.head_dim
        kv_size = attn.num_kv_heads * attn.head_dim
        k_processed = params.qkv_input.narrow(1, q_size, kv_size).view(
            params.num_tokens,
            attn.num_kv_heads,
            attn.head_dim,
        )
        v_processed = params.qkv_input.narrow(1, q_size + kv_size, kv_size).view(
            params.num_tokens,
            attn.num_kv_heads,
            attn.head_dim,
        )

        block_tables = block_tables[: params.batch_size]
        blocks_per_sequence = block_tables.size(-1)
        page_table_indptr = (
            torch.arange(
                params.batch_size + 1,
                dtype=torch.int32,
                device=block_tables.device,
            )
            * blocks_per_sequence
        )
        prefix_lens = (
            params.sequence_lengths[: params.batch_size]
            - params.context_lengths[: params.batch_size]
        )

        if meta.kv_cache_manager is None:
            raise RuntimeError("Custom-mask TRT-LLM attention requires a KV cache manager.")
        kv_cache = meta.kv_cache_manager.get_buffers(
            attn.layer_idx,
            kv_layout=meta.kv_layout,
        )
        if kv_cache is None:
            raise RuntimeError("Custom-mask TRT-LLM attention requires a KV cache buffer.")

        # TRTLLM block tables index a flat physical-page pool, while get_buffers()
        # groups K and V into each logical page. Convert the physical K-page IDs
        # to indices in that packed view using its actual storage strides.
        physical_pages_per_logical_page = kv_cache.stride(0) // kv_cache.stride(1)
        page_table_indices = torch.div(
            block_tables[:, 0, :],
            physical_pages_per_logical_page,
            rounding_mode="floor",
        ).reshape(-1)

        from ..triton_prefill import triton_prefill_with_custom_mask

        triton_prefill_with_custom_mask(
            q=q_processed,
            k=k_processed,
            v=v_processed,
            output=params.context_buf,
            qo_indptr=cu_q_seqlens,
            kv_cache=kv_cache,
            prefix_lens=prefix_lens,
            page_table_indptr=page_table_indptr,
            page_table_indices=page_table_indices,
            page_size=params.tokens_per_block,
            custom_mask=fwd.attention_mask_data.flatten().contiguous(),
            sm_scale=bmm1_scale,
            window_left=window_left,
        )
