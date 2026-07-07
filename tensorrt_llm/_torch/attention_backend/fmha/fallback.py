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

from typing import TYPE_CHECKING, Optional

import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs
from tensorrt_llm.bindings.internal import thop

from .interface import Fmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata


# ``AttentionForwardArgs`` fields that this backend does not consume.
# Sync test (test_attention_op_sync.py) requires every other field to map to a
# kwarg name, a @property on the dataclass, or a field that some @property
# transitively reads; entries here are exempt.
_THOP_EXCLUDED_FIELDS: frozenset = frozenset(
    {
        "topk_indices",  # DSA-only
        "attention_mask_data",  # custom-mask code path
        "out_scale_sf",  # promoted into ``out_scale`` in ``TrtllmAttention.forward`` for NVFP4 path
        "skip_mla_rope_generation",  # handled in ``TrtllmAttention.forward`` for the test-only MLA path
    }
)

# ``thop.attention`` kwargs hard-wired to a literal at the call site (no
# rich object owns them). Sync test enforces both the kwarg name and the
# literal value.
_THOP_LITERALS: dict = {}


class FallbackFmha(Fmha):
    """Fallback FMHA implementation using the fused TRT-LLM thop attention op."""

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        attn = self.attn

        # Every kwarg sources from ``attn`` / ``metadata`` / ``forward_args``
        # (with ``forward_args.sparse_prediction`` for sparse-attn inputs),
        # or a literal allowlisted in ``_THOP_LITERALS``.
        # ``test_attention_op_sync.py`` enforces this statically.
        thop.attention(
            q=q,
            k=k,
            v=v,
            output=forward_args.output,
            output_sf=forward_args.output_sf,
            workspace_=metadata.effective_workspace,
            # --- Per-step batch state (TrtllmAttentionMetadata) ---
            sequence_length=metadata.kv_lens_cuda_runtime,
            host_past_key_value_lengths=metadata.kv_lens_runtime,
            host_total_kv_lens=metadata.host_total_kv_lens,
            context_lengths=metadata.prompt_lens_cuda_runtime,
            host_context_lengths=metadata.prompt_lens_cpu_runtime,
            host_request_types=metadata.host_request_types_runtime,
            max_context_q_len_override=metadata.max_context_q_len_override,
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            cache_indirection=metadata.cache_indirection,
            block_ids_per_seq=metadata.block_ids_per_seq,
            tokens_per_block=metadata.tokens_per_block,
            max_num_requests=metadata.max_num_requests,
            beam_width=metadata.effective_beam_width,
            use_paged_context_fmha=metadata.use_paged_context_fmha,
            helix_position_offsets=metadata.helix_position_offsets,
            helix_is_inactive_rank=metadata.helix_is_inactive_rank,
            is_spec_decoding_enabled=metadata.is_spec_decoding_enabled,
            use_spec_decoding=metadata.use_spec_decoding,
            is_spec_dec_tree=metadata.is_spec_dec_tree,
            spec_decoding_generation_lengths=metadata.spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp=metadata.spec_decoding_position_offsets_for_cpp,
            spec_decoding_packed_mask=metadata.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=metadata.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=metadata.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=metadata.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=metadata.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=metadata.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=metadata.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=metadata.flash_mla_num_splits,
            num_contexts=metadata.num_contexts,
            num_ctx_tokens=metadata.num_ctx_tokens,
            max_context_length=metadata.max_context_length,
            max_seq_len=metadata.max_seq_len,
            trtllm_gen_jit_warmup=metadata.trtllm_gen_jit_warmup,
            is_cross=metadata.is_cross,
            # --- Per-call (AttentionForwardArgs) ---
            out_scale=forward_args.out_scale,
            kv_scale_orig_quant=forward_args.kv_scale_orig_quant,
            kv_scale_quant_orig=forward_args.kv_scale_quant_orig,
            latent_cache=forward_args.latent_cache,
            q_pe=forward_args.q_pe,
            attention_sinks=forward_args.attention_sinks,
            mask_type=forward_args.mask_type,
            attention_input_type=int(forward_args.attention_input_type),
            attention_window_size=forward_args.attention_window_size,
            chunked_prefill_buffer_batch_size=forward_args.chunked_prefill_buffer_batch_size,
            mrope_rotary_cos_sin=forward_args.mrope_rotary_cos_sin,
            mrope_position_deltas=forward_args.mrope_position_deltas,
            softmax_stats_tensor=forward_args.softmax_stats_tensor,
            cu_q_seqlens=forward_args.cu_q_seqlens,
            cu_kv_seqlens=forward_args.cu_kv_seqlens,
            fmha_scheduler_counter=forward_args.fmha_scheduler_counter,
            mla_bmm1_scale=forward_args.mla_bmm1_scale,
            mla_bmm2_scale=forward_args.mla_bmm2_scale,
            quant_q_buffer=forward_args.quant_q_buffer,
            quant_scale_qkv=forward_args.quant_scale_qkv,
            sage_attn_num_elts_per_blk_q=forward_args.sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=forward_args.sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=forward_args.sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=forward_args.sage_attn_qk_int8,
            is_fused_qkv=forward_args.is_fused_qkv,
            update_kv_cache=forward_args.update_kv_cache,
            cross_kv=forward_args.cross_kv,
            relative_attention_bias=forward_args.relative_attention_bias,
            relative_attention_max_distance=forward_args.relative_attention_max_distance,
            skip_softmax_threshold_scale_factor_prefill=forward_args.skip_softmax_kernel_params.threshold_scale_factor_prefill,
            skip_softmax_threshold_scale_factor_decode=forward_args.skip_softmax_kernel_params.threshold_scale_factor_decode,
            # --- Module config (TrtllmAttention) ---
            rotary_inv_freq=attn.rotary_inv_freq,
            rotary_cos_sin=attn.rotary_cos_sin,
            predicted_tokens_per_seq=attn.predicted_tokens_per_seq,
            local_layer_idx=attn.local_layer_idx,
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            quant_mode=attn.quant_mode,
            q_scaling=attn.q_scaling,
            position_embedding_type=attn.position_embedding_type,
            rope_dim=attn.rope_dim,
            rope_base=attn.rope_base,
            rope_scale_type=attn.rope_scale_type,
            rope_scale=attn.rope_scale,
            rope_short_m_scale=attn.rope_short_m_scale,
            rope_long_m_scale=attn.rope_long_m_scale,
            rope_max_positions=attn.rope_max_positions,
            rope_original_max_positions=attn.rope_original_max_positions,
            is_mla_enable=attn.is_mla_enable,
            q_lora_rank=attn.q_lora_rank,
            kv_lora_rank=attn.kv_lora_rank,
            qk_nope_head_dim=attn.qk_nope_head_dim,
            qk_rope_head_dim=attn.qk_rope_head_dim,
            v_head_dim=attn.v_head_dim,
            rope_append=attn.rope_append,
            attention_chunk_size=attn.attention_chunk_size,
            skip_softmax_stat=attn.skip_softmax_stat,
            # --- Sparse-specific (AttentionForwardArgs.sparse_prediction) ---
            sparse_kv_indices=forward_args.sparse_prediction.sparse_kv_indices,
            sparse_kv_offsets=forward_args.sparse_prediction.sparse_kv_offsets,
            sparse_attn_indices=forward_args.sparse_prediction.sparse_attn_indices,
            sparse_attn_offsets=forward_args.sparse_prediction.sparse_attn_offsets,
            sparse_attn_indices_block_size=forward_args.sparse_prediction.sparse_attn_indices_block_size,
            sparse_mla_topk_lens=forward_args.sparse_prediction.sparse_mla_topk_lens,
            compressed_kv_cache_pool_ptr=forward_args.sparse_prediction.compressed_kv_cache_pool_ptr,
        )
