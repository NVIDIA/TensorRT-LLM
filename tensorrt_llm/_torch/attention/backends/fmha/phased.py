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

from typing import TYPE_CHECKING, Optional, cast

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
)

from .interface import Fmha, FmhaParams

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class PhasedFmha(Fmha):
    """FMHA helper for paged-KV libraries that split work by request phase."""

    REQUIRES_PAGED_KV = True
    # Set by libraries that address the paged KV pool tensor directly. Resolving it
    # costs a slice + reshape, so it is opt-in.
    NEEDS_KV_POOL = False

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self.kv_factor = 1 if attn.is_mla_enable else 2
        # Must match the buffer `TrtllmAttention.create_output` allocates, which is
        # what `_build_params` takes this view over.
        self.generation_out_head_size = attn.out_head_size(is_gen_only=True)
        self.context_out_head_size = attn.out_head_size(is_gen_only=False)

    def _get_total_num_blocks(
        self,
        meta: "TrtllmAttentionMetadata",
    ) -> int:
        kv_cache_manager = meta.kv_cache_manager
        if kv_cache_manager is None:
            return 0

        get_page_index_upper_bound = getattr(
            getattr(kv_cache_manager, "impl", None),
            "get_page_index_upper_bound",
            None,
        )
        if get_page_index_upper_bound is not None:
            # KVCacheManagerV2 exposes an already-flattened page-index bound,
            # unlike the legacy logical block count.
            return int(kv_cache_manager.blocks_in_primary_pool)

        blocks_in_primary_pool = getattr(kv_cache_manager, "blocks_in_primary_pool", None)
        if blocks_in_primary_pool is None:
            blocks_per_window = getattr(kv_cache_manager, "blocks_per_window", None)
            if blocks_per_window:
                blocks_in_primary_pool = max(
                    int(primary) for primary, _ in blocks_per_window.values()
                )
        if blocks_in_primary_pool is None:
            return 0
        return int(blocks_in_primary_pool) * kv_cache_manager.num_local_layers * self.kv_factor

    def get_fp8_context_fmha(
        self,
        q: torch.Tensor,
        output: torch.Tensor,
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        is_gen_only: bool,
    ) -> bool:
        return False

    def _build_params(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
        *,
        fp8_context_fmha: bool,
    ) -> FmhaParams:
        """Build the common Python parameters used for sizing and execution."""
        attn = self.attn
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")

        num_tokens = q.size(0)
        is_gen_only = forward_args.attention_input_type == AttentionInputType.generation_only
        out_head_size = self.generation_out_head_size if is_gen_only else self.context_out_head_size
        attention_window_size = forward_args.attention_window_size
        cache_indirection = metadata.cache_indirection
        max_attention_window_size = (
            attention_window_size
            if metadata.beam_width == 1
            else (
                cache_indirection.size(2)
                if cache_indirection is not None
                else attention_window_size
            )
        )
        tokens_per_block = (
            metadata.tokens_per_block if metadata.tokens_per_block is not None else 64
        )

        kv_pool = None
        if self.NEEDS_KV_POOL and metadata.kv_cache_manager is not None:
            kv_pool = metadata.kv_cache_manager.get_buffers(attn.layer_idx)

        return FmhaParams(
            workspace=workspace,
            fwd=forward_args,
            # Python-only back-references for backends that need layer/metadata state
            # the flat schema does not carry (e.g. the Triton custom-mask backend).
            attn=attn,
            meta=metadata,
            qkv_or_q=q,
            k=k,
            v=v,
            output=output.view(num_tokens, attn.num_heads, out_head_size),
            sequence_length=metadata.kv_lens_cuda_runtime,
            context_lengths=metadata.prompt_lens_cuda_runtime,
            host_past_key_value_lengths=metadata.kv_lens_runtime,
            host_context_lengths=metadata.prompt_lens_cpu_runtime,
            # Layer config
            layer_idx=attn.layer_idx,
            local_layer_idx=attn.get_local_layer_idx(metadata),
            num_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_size=attn.head_dim,
            q_scaling=attn.q_scaling,
            quant_mode=attn.quant_mode,
            position_embedding_type=attn.position_embedding_type,
            predicted_tokens_per_seq=attn.predicted_tokens_per_seq,
            attention_chunk_size=attn.attention_chunk_size,
            has_fp8_kv_cache=bool(getattr(attn, "has_fp8_kv_cache", False)),
            rope_params=attn.rope_params,
            rotary_inv_freq=attn.rotary_inv_freq,
            rotary_cos_sin=attn.rotary_cos_sin,
            is_mla_enable=attn.is_mla_enable,
            q_lora_rank=attn.q_lora_rank,
            kv_lora_rank=attn.kv_lora_rank or 0,
            qk_nope_head_dim=attn.qk_nope_head_dim or 0,
            qk_rope_head_dim=attn.qk_rope_head_dim or 0,
            v_head_dim=attn.v_head_dim,
            rope_append=attn.rope_append,
            # Runtime metadata
            kv_cache_block_offsets=metadata.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
            cache_indirection=cache_indirection,
            max_context_q_len_override=metadata.max_context_q_len_override,
            block_ids_per_seq=metadata.block_ids_per_seq,
            helix_position_offsets=metadata.helix_position_offsets,
            helix_is_inactive_rank=metadata.helix_is_inactive_rank,
            spec_decoding_generation_lengths=metadata.spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp=metadata.spec_decoding_position_offsets_for_cpp,
            spec_decoding_packed_mask=metadata.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=metadata.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=metadata.spec_decoding_bl_tree_mask,
            spec_bl_tree_first_sparse_mask_offset_kv=metadata.spec_bl_tree_first_sparse_mask_offset_kv,
            flash_mla_tile_scheduler_metadata=metadata.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=metadata.flash_mla_num_splits,
            trtllm_gen_jit_warmup=bool(metadata.trtllm_gen_jit_warmup),
            is_cross=metadata.is_cross,
            num_sparse_topk=metadata.num_sparse_topk or 0,
            use_paged_context_fmha=metadata.use_paged_context_fmha,
            paged_context_fmha=metadata.use_paged_context_fmha,
            beam_width=metadata.beam_width,
            max_num_requests=metadata.max_num_requests,
            max_num_sequences=metadata.max_num_sequences or metadata.max_num_requests,
            max_context_length=metadata.max_context_length,
            max_seq_len=metadata.max_seq_len,
            is_spec_decoding_enabled=metadata.is_spec_decoding_enabled,
            use_spec_decoding=metadata.use_spec_decoding,
            is_spec_dec_tree=metadata.is_spec_dec_tree,
            force_prepare_spec_dec_tree_mask=metadata.force_prepare_spec_dec_tree_mask,
            spec_decoding_target_max_draft_tokens=metadata.max_total_draft_tokens,
            # Shared phase state
            max_attention_window_size=max_attention_window_size,
            cyclic_attention_window_size=attention_window_size,
            tokens_per_block=tokens_per_block,
            fp8_context_fmha=fp8_context_fmha,
            kv_factor=self.kv_factor,
            total_num_blocks=self._get_total_num_blocks(metadata),
            kv_pool=kv_pool,
            num_tokens=num_tokens,
        )

    def prepare_workspace(
        self,
        params: FmhaParams,
        metadata: "TrtllmAttentionMetadata",
    ) -> None:
        pass

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        attn = self.attn
        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")
        if self.REQUIRES_PAGED_KV and metadata.kv_cache_block_offsets is None:
            raise RuntimeError(f"{type(self).__name__} requires paged KV cache.")

        workspace = cast(torch.Tensor, metadata.effective_workspace)

        num_tokens = q.size(0)
        attention_input_type = forward_args.attention_input_type
        is_gen_only = attention_input_type == AttentionInputType.generation_only

        num_contexts = metadata.num_contexts
        num_ctx_tokens = metadata.num_ctx_tokens
        num_generations = metadata.num_generations
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid FMHA token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}, attention_input_type={attention_input_type}."
            )

        fp8_context_fmha = self.get_fp8_context_fmha(q, output, metadata, forward_args, is_gen_only)
        params = self._build_params(
            q,
            k,
            v,
            metadata,
            forward_args,
            workspace,
            fp8_context_fmha=fp8_context_fmha,
        )
        self.prepare_workspace(params, metadata)

        out_tensor = cast(torch.Tensor, params.output)

        sequence_length = metadata.kv_lens_cuda_runtime
        host_past_key_value_lengths = metadata.kv_lens_runtime
        host_total_kv_lens = metadata.host_total_kv_lens

        if num_contexts > 0 and attention_input_type != AttentionInputType.generation_only:
            seq_offset = 0
            token_offset = 0
            num_seqs = num_contexts

            context_lengths = metadata.prompt_lens_cuda_runtime
            host_context_lengths = metadata.prompt_lens_cpu_runtime
            max_context_q_len = int(host_context_lengths[seq_offset : seq_offset + num_seqs].max())
            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )
            # Encoder CUDA graphs capture a padded context extent; the override widens both
            # the q length and the past-kv length so the captured shapes stay valid.
            override = metadata.max_context_q_len_override
            if override is not None:
                override = int(override)
                if override < max_context_q_len:
                    raise ValueError(
                        f"max_context_q_len_override ({override}) must be >= computed max "
                        f"context q length ({max_context_q_len})."
                    )
                if override < max_past_kv_len:
                    raise ValueError(
                        f"max_context_q_len_override ({override}) must be >= computed max "
                        f"past kv length ({max_past_kv_len})."
                    )
                max_context_q_len = override
                max_past_kv_len = override

            params.qkv_or_q = q[token_offset : token_offset + num_ctx_tokens]
            params.k = None if k is None else k[token_offset : token_offset + num_ctx_tokens]
            params.v = None if v is None else v[token_offset : token_offset + num_ctx_tokens]
            params.output = out_tensor[token_offset : token_offset + num_ctx_tokens]
            params.sequence_length = sequence_length[seq_offset:]
            params.context_lengths = context_lengths[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_ctx_tokens
            params.seq_offset = seq_offset
            params.token_offset = token_offset
            params.input_seq_length = max_context_q_len
            params.num_seqs = num_seqs
            params.total_kv_len = int(host_total_kv_lens[0])
            if attn.is_mla_enable:
                self.run_mla_context(params)
            else:
                self.run_context(params)

        if num_generations > 0 and attention_input_type != AttentionInputType.context_only:
            seq_offset = num_contexts
            token_offset = 0 if is_gen_only else num_ctx_tokens
            num_seqs = num_generations

            max_past_kv_len = int(
                host_past_key_value_lengths[seq_offset : seq_offset + num_seqs].max()
            )
            input_seq_length = num_gen_tokens // num_seqs if num_seqs > 0 else 1

            # The spec-decoding tensors are already on `params` from the metadata; only
            # the position offsets need reshaping. Do not gate on
            # `predicted_tokens_per_seq`: the native side asks for them whenever
            # `use_spec_decoding` is set, which a non-MLA layer reaches with the default
            # `predicted_tokens_per_seq` of 1.
            position_offsets_for_cpp = metadata.spec_decoding_position_offsets_for_cpp
            if position_offsets_for_cpp is not None and position_offsets_for_cpp.dim() == 1:
                position_offsets_for_cpp = position_offsets_for_cpp.view(
                    metadata.max_num_requests, -1
                )

            params.qkv_or_q = q[token_offset : token_offset + num_gen_tokens]
            params.k = None if k is None else k[token_offset : token_offset + num_gen_tokens]
            params.v = None if v is None else v[token_offset : token_offset + num_gen_tokens]
            params.output = out_tensor[token_offset : token_offset + num_gen_tokens]
            params.sequence_length = sequence_length[seq_offset:]
            params.context_lengths = metadata.prompt_lens_cuda_runtime[seq_offset:]
            params.max_past_kv_length = max_past_kv_len
            params.num_tokens = num_gen_tokens
            params.seq_offset = seq_offset
            params.token_offset = token_offset
            params.input_seq_length = input_seq_length
            params.num_seqs = num_seqs
            params.num_requests = num_seqs // metadata.beam_width
            params.total_kv_len = int(host_total_kv_lens[1])
            params.spec_decoding_position_offsets_for_cpp = position_offsets_for_cpp
            if attn.is_mla_enable:
                self.run_mla_generation(params)
            else:
                # The custom mask covers only the context portion of a mixed batch.
                if (
                    not metadata.is_cross
                    and params.fwd.attention_mask == CustomAttentionMask.CUSTOM
                ):
                    params.fwd.attention_mask = PredefinedAttentionMask.CAUSAL
                    params.fwd.attention_mask_data = None
                self.run_generation(params)

    def run_context(self, params: FmhaParams) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support context attention.")

    def run_generation(self, params: FmhaParams) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support generation attention.")

    def run_mla_context(self, params: FmhaParams) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support MLA context attention.")

    def run_mla_generation(self, params: FmhaParams) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support MLA generation attention."
        )
