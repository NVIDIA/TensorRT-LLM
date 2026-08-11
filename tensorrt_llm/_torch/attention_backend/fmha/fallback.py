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

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm.bindings.internal import thop

from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class FallbackFmha(PhasedFmha):
    """Fallback FMHA implementation over the phased TRT-LLM thop ops."""

    REQUIRES_PAGED_KV = False

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self.generation_out_head_size = self.context_out_head_size

    @staticmethod
    def run_auto_deploy_mha(
        qkv_or_q: torch.Tensor,
        output: torch.Tensor,
        workspace: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_total_kv_lens: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_pool_pointers: torch.Tensor,
        host_kv_cache_pool_mapping: torch.Tensor,
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        rotary_cos_sin: Optional[torch.Tensor],
        attention_sinks: Optional[torch.Tensor],
        spec_decoding_generation_lengths: Optional[torch.Tensor],
        spec_decoding_position_offsets_for_cpp: Optional[torch.Tensor],
        spec_decoding_packed_mask: Optional[torch.Tensor],
        num_contexts: int,
        num_ctx_tokens: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: int,
        max_num_requests: int,
        max_context_length: int,
        max_seq_len: int,
        attention_window_size: int,
        mask_type: int,
        quant_mode: int,
        q_scaling: float,
        position_embedding_type: int,
        rope_dim: int,
        is_spec_decoding_enabled: bool,
        use_spec_decoding: bool,
    ) -> None:
        if workspace is None:
            raise RuntimeError("FallbackFmha requires workspace.")

        num_tokens = qkv_or_q.size(0)
        num_generations = host_context_lengths.size(0) - num_contexts
        num_gen_tokens = num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid FMHA token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}."
            )

        thop_params = thop.FmhaParams()
        thop_params.layer_idx = 0
        thop_params.num_heads = num_heads
        thop_params.num_kv_heads = num_kv_heads
        thop_params.head_size = head_size
        thop_params.tokens_per_block = tokens_per_block
        thop_params.max_num_requests = max_num_requests
        thop_params.max_context_length = max_context_length
        thop_params.max_seq_len = max_seq_len
        thop_params.attention_window_size = attention_window_size
        thop_params.beam_width = 1
        thop_params.mask_type = mask_type
        thop_params.quant_mode = quant_mode
        thop_params.q_scaling = q_scaling
        thop_params.position_embedding_type = position_embedding_type
        thop_params.rotary_embedding_dim = rope_dim
        thop_params.rotary_embedding_base = 10000.0
        thop_params.rotary_embedding_scale_type = 0
        thop_params.rotary_embedding_scale = 1.0
        thop_params.rotary_embedding_short_mscale = 1.0
        thop_params.rotary_embedding_long_mscale = 1.0
        thop_params.rotary_embedding_max_positions = max_context_length
        thop_params.rotary_embedding_original_max_positions = max_context_length
        thop_params.paged_context_fmha = True
        thop_params.is_mla_enable = False
        thop_params.chunk_prefill_buffer_batch_size = max_num_requests
        thop_params.attention_chunk_size = None
        thop_params.is_spec_decoding_enabled = is_spec_decoding_enabled
        thop_params.use_spec_decoding = use_spec_decoding
        thop_params.is_spec_dec_tree = False
        thop_params.spec_decoding_target_max_draft_tokens = None
        thop_params.force_prepare_spec_dec_tree_mask = False
        thop_params.sage_attn_num_elts_per_blk_q = 0
        thop_params.sage_attn_num_elts_per_blk_k = 0
        thop_params.sage_attn_num_elts_per_blk_v = 0
        thop_params.sage_attn_qk_int8 = False
        thop_params.max_distance = 0
        thop_params.skip_softmax_threshold_scale_factor_prefill = 0.0
        thop_params.skip_softmax_threshold_scale_factor_decode = 0.0
        thop_params.predicted_tokens_per_seq = 1
        thop_params.workspace = workspace
        thop_params.output = output
        thop_params.qkv_or_q = qkv_or_q
        thop_params.sequence_length = sequence_length
        thop_params.host_past_key_value_lengths = host_past_key_value_lengths
        thop_params.context_lengths = context_lengths
        thop_params.host_context_lengths = host_context_lengths
        thop_params.kv_cache_block_offsets = kv_cache_block_offsets
        thop_params.host_kv_cache_pool_pointers = host_kv_cache_pool_pointers
        thop_params.host_kv_cache_pool_mapping = host_kv_cache_pool_mapping
        thop_params.kv_scale_orig_quant = kv_scale_orig_quant
        thop_params.kv_scale_quant_orig = kv_scale_quant_orig
        thop_params.out_scale = out_scale
        thop_params.rotary_cos_sin = rotary_cos_sin
        thop_params.attention_sinks = attention_sinks
        thop_params.spec_decoding_generation_lengths = spec_decoding_generation_lengths
        thop_params.spec_decoding_position_offsets_for_cpp = spec_decoding_position_offsets_for_cpp
        thop_params.spec_decoding_packed_mask = spec_decoding_packed_mask
        thop_params.sparse_attn_indices_block_size = 1
        thop_params.num_sparse_topk = 0

        workspace_size = thop.get_attention_workspace_size(
            thop_params,
            num_tokens,
            attention_window_size,
            num_gen_tokens,
            kv_cache_block_offsets.size(-1),
            int(host_total_kv_lens[0]),
        )
        if workspace.numel() < workspace_size:
            workspace.resize_(workspace_size)

        if num_contexts > 0:
            thop_params.qkv_or_q = qkv_or_q[:num_ctx_tokens]
            thop_params.output = output[:num_ctx_tokens]
            thop_params.seq_offset = 0
            thop_params.num_seqs = num_contexts
            thop_params.token_offset = 0
            thop_params.num_tokens = num_ctx_tokens
            thop_params.total_kv_len = int(host_total_kv_lens[0])
            thop.run_context(thop_params)

        if num_generations > 0:
            thop_params.qkv_or_q = qkv_or_q[num_ctx_tokens : num_ctx_tokens + num_gen_tokens]
            thop_params.output = output[num_ctx_tokens : num_ctx_tokens + num_gen_tokens]
            thop_params.seq_offset = num_contexts
            thop_params.num_seqs = num_generations
            thop_params.token_offset = 0
            thop_params.num_tokens = num_gen_tokens
            thop_params.total_kv_len = int(host_total_kv_lens[1])
            thop.run_generation(thop_params)

    def _fill_static_thop_params(
        self,
        tp,
        meta: "TrtllmAttentionMetadata",
        fa: AttentionForwardArgs,
    ) -> None:
        attn = self.attn
        output = fa.output
        if output is None:
            raise RuntimeError("FallbackFmha requires output.")

        sp = fa.sparse_runtime_params
        skip = sp
        tp.layer_idx = attn.local_layer_idx
        tp.num_heads = attn.num_heads
        tp.num_kv_heads = attn.num_kv_heads
        tp.head_size = attn.head_dim
        tp.tokens_per_block = meta.tokens_per_block or 0
        tp.max_num_requests = meta.max_num_requests
        tp.max_context_length = meta.max_context_length
        tp.max_seq_len = meta.max_seq_len
        tp.attention_window_size = fa.attention_window_size
        tp.beam_width = meta.effective_beam_width
        tp.mask_type = fa.mask_type
        tp.quant_mode = attn.quant_mode
        tp.q_scaling = attn.q_scaling
        tp.position_embedding_type = attn.position_embedding_type
        tp.rotary_embedding_dim = attn.rope_dim
        tp.rotary_embedding_base = attn.rope_base
        tp.rotary_embedding_scale_type = attn.rope_scale_type
        tp.rotary_embedding_scale = attn.rope_scale
        tp.rotary_embedding_short_mscale = attn.rope_short_m_scale
        tp.rotary_embedding_long_mscale = attn.rope_long_m_scale
        tp.rotary_embedding_max_positions = attn.rope_max_positions
        tp.rotary_embedding_original_max_positions = attn.rope_original_max_positions
        tp.paged_context_fmha = meta.use_paged_context_fmha
        tp.is_mla_enable = attn.is_mla_enable
        tp.chunk_prefill_buffer_batch_size = fa.chunked_prefill_buffer_batch_size or 1
        tp.q_lora_rank = attn.q_lora_rank
        tp.kv_lora_rank = attn.kv_lora_rank
        tp.qk_nope_head_dim = attn.qk_nope_head_dim
        tp.qk_rope_head_dim = attn.qk_rope_head_dim
        tp.v_head_dim = attn.v_head_dim
        tp.rope_append = attn.rope_append
        tp.attention_chunk_size = attn.attention_chunk_size
        tp.is_spec_decoding_enabled = meta.is_spec_decoding_enabled
        tp.use_spec_decoding = meta.use_spec_decoding
        tp.is_spec_dec_tree = meta.is_spec_dec_tree
        tp.spec_decoding_target_max_draft_tokens = meta.max_total_draft_tokens
        tp.force_prepare_spec_dec_tree_mask = meta.force_prepare_spec_dec_tree_mask
        tp.sage_attn_num_elts_per_blk_q = fa.sage_attn_num_elts_per_blk_q
        tp.sage_attn_num_elts_per_blk_k = fa.sage_attn_num_elts_per_blk_k
        tp.sage_attn_num_elts_per_blk_v = fa.sage_attn_num_elts_per_blk_v
        tp.sage_attn_qk_int8 = fa.sage_attn_qk_int8
        tp.max_distance = fa.relative_attention_max_distance
        tp.skip_softmax_threshold_scale_factor_prefill = skip.threshold_scale_factor_prefill or 0.0
        tp.skip_softmax_threshold_scale_factor_decode = skip.threshold_scale_factor_decode or 0.0

    def _build_common_thop_params(
        self,
        qkv_input: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor,
        meta: "TrtllmAttentionMetadata",
        fa: AttentionForwardArgs,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        attn = self.attn
        sp = fa.sparse_runtime_params
        tp = thop.FmhaParams()
        self._fill_static_thop_params(tp, meta, fa)
        tp.predicted_tokens_per_seq = attn.predicted_tokens_per_seq
        tp.workspace = workspace
        tp.output = output
        tp.output_sf = fa.output_sf
        tp.qkv_or_q = qkv_input
        tp.k = k
        tp.v = v
        tp.sequence_length = meta.kv_lens_cuda_runtime
        tp.host_past_key_value_lengths = meta.kv_lens_runtime
        tp.context_lengths = meta.prompt_lens_cuda_runtime
        tp.host_context_lengths = meta.prompt_lens_cpu_runtime
        tp.max_context_q_len_override = meta.max_context_q_len_override
        tp.kv_cache_block_offsets = meta.kv_cache_block_offsets
        tp.host_kv_cache_pool_pointers = meta.host_kv_cache_pool_pointers
        tp.host_kv_cache_pool_mapping = meta.host_kv_cache_pool_mapping
        tp.cache_indirection = meta.cache_indirection
        tp.kv_scale_orig_quant = fa.kv_scale_orig_quant
        tp.kv_scale_quant_orig = fa.kv_scale_quant_orig
        tp.out_scale = fa.out_scale
        tp.rotary_inv_freq = attn.rotary_inv_freq
        tp.rotary_cos_sin = attn.rotary_cos_sin
        tp.latent_cache = fa.latent_cache
        tp.q_pe = fa.q_pe
        tp.block_ids_per_seq = meta.block_ids_per_seq
        tp.mrope_rotary_cos_sin = fa.mrope_rotary_cos_sin
        tp.mrope_position_deltas = fa.mrope_position_deltas
        tp.helix_position_offsets = meta.helix_position_offsets
        tp.helix_is_inactive_rank = meta.helix_is_inactive_rank
        tp.softmax_stats_tensor = fa.softmax_stats_tensor
        tp.spec_decoding_generation_lengths = meta.spec_decoding_generation_lengths
        tp.spec_decoding_position_offsets_for_cpp = meta.spec_decoding_position_offsets_for_cpp
        tp.spec_decoding_packed_mask = meta.spec_decoding_packed_mask
        tp.spec_decoding_bl_tree_mask_offset = meta.spec_decoding_bl_tree_mask_offset
        tp.spec_decoding_bl_tree_mask = meta.spec_decoding_bl_tree_mask
        tp.spec_bl_tree_first_sparse_mask_offset_kv = meta.spec_bl_tree_first_sparse_mask_offset_kv
        tp.attention_sinks = fa.attention_sinks
        tp.sparse_kv_indices = sp.sparse_kv_indices
        tp.sparse_kv_offsets = sp.sparse_kv_offsets
        tp.sparse_attn_indices = sp.sparse_attn_indices
        tp.sparse_attn_offsets = sp.sparse_attn_offsets
        tp.sparse_attn_indices_block_size = sp.sparse_attn_indices_block_size
        tp.num_sparse_topk = meta.num_sparse_topk or 0
        tp.sparse_attn_kv_lens = sp.sparse_attn_kv_lens
        tp.cu_q_seqlens = fa.cu_q_seqlens
        tp.cu_kv_seqlens = fa.cu_kv_seqlens
        tp.fmha_scheduler_counter = fa.fmha_scheduler_counter
        tp.mla_bmm1_scale = fa.mla_bmm1_scale
        tp.mla_bmm2_scale = fa.mla_bmm2_scale
        tp.quant_q_buffer = fa.quant_q_buffer
        tp.flash_mla_tile_scheduler_metadata = meta.flash_mla_tile_scheduler_metadata
        tp.flash_mla_num_splits = meta.flash_mla_num_splits
        tp.trtllm_gen_jit_warmup = bool(meta.trtllm_gen_jit_warmup)
        tp.aux_kv_cache_pool_ptr = sp.aux_kv_cache_pool_ptr
        tp.is_cross = meta.is_cross
        tp.cross_kv = fa.cross_kv
        tp.relative_attention_bias = fa.relative_attention_bias
        tp.quant_scale_qkv = fa.quant_scale_qkv
        tp.dsv4_inv_rope_cos_sin_cache = fa.dsv4_inv_rope_cos_sin_cache
        tp.enable_dsv4_epilogue_fusion = fa.enable_dsv4_epilogue_fusion
        return tp

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        fa = forward_args
        meta = metadata
        output = fa.output
        if output is None:
            raise RuntimeError("FallbackFmha requires output.")
        tp = self._build_common_thop_params(q, output, workspace, meta, fa, k, v)

        num_tokens = q.size(0)
        is_gen_only = fa.attention_input_type == AttentionInputType.generation_only
        num_ctx_tokens = meta.num_ctx_tokens
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - num_ctx_tokens
        ctx_total_kv_len = int(meta.host_total_kv_lens[0])

        beam_width = meta.effective_beam_width
        cache_indirection = meta.cache_indirection
        max_attention_window_size = (
            fa.attention_window_size
            if beam_width == 1
            else (
                cache_indirection.size(2)
                if cache_indirection is not None
                else fa.attention_window_size
            )
        )
        kv_cache_block_offsets = meta.kv_cache_block_offsets
        use_kv_cache = kv_cache_block_offsets is not None
        max_blocks_per_sequence = kv_cache_block_offsets.size(-1) if use_kv_cache else 0
        workspace_size = thop.get_attention_workspace_size(
            tp,
            num_tokens,
            max_attention_window_size,
            num_gen_tokens,
            max_blocks_per_sequence,
            ctx_total_kv_len,
        )
        if workspace is not None and workspace.numel() < workspace_size:
            workspace.resize_(workspace_size)

    def _build_thop_params(self, params: FmhaParams, is_context: bool):
        meta, fa = params.meta, params.fwd
        qkv_input = params.qkv_or_q
        if qkv_input is None:
            raise RuntimeError("FallbackFmha requires qkv_or_q.")
        output = params.output
        if output is None:
            raise RuntimeError("FallbackFmha requires output.")

        num_contexts = meta.num_contexts
        num_seqs = (
            num_contexts if is_context else meta.prompt_lens_cpu_runtime.size(0) - num_contexts
        )
        total_kv_len = int(meta.host_total_kv_lens[0 if is_context else 1])

        tp = self._build_common_thop_params(
            qkv_input, output, params.workspace, meta, fa, params.k, params.v
        )

        tp.seq_offset = params.seq_offset
        tp.num_seqs = num_seqs
        tp.token_offset = 0
        tp.num_tokens = params.num_tokens
        tp.total_kv_len = total_kv_len
        return tp

    def run_context(self, params: FmhaParams) -> None:
        tp = self._build_thop_params(params, is_context=True)
        thop.run_context(tp)

    def run_mla_context(self, params: FmhaParams) -> None:
        tp = self._build_thop_params(params, is_context=True)
        thop.run_context(tp)

    def run_generation(self, params: FmhaParams) -> None:
        tp = self._build_thop_params(params, is_context=False)
        thop.run_generation(tp)

    def run_mla_generation(self, params: FmhaParams) -> None:
        tp = self._build_thop_params(params, is_context=False)
        thop.run_mla_generation(tp)
