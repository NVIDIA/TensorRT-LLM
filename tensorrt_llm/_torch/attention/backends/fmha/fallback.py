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

from typing import TYPE_CHECKING, ClassVar, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    CustomAttentionMask,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.logger import logger

from .interface import Fmha, FmhaPhase

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


# ``AttentionForwardArgs`` fields that this backend does not consume.
# Sync test (test_attention_op_sync.py) requires every other field to map to a
# kwarg name, a @property on the dataclass, or a field that some @property
# transitively reads; entries here are exempt.
_THOP_EXCLUDED_FIELDS: frozenset = frozenset(
    {
        "sparse_backend_args",  # consumed by sparse prediction before the attention op
        "block_sparse_inputs",  # consumed by the selected block-sparse FMHA
        "attention_mask_data",  # custom-mask code path
        "out_scale_sf",  # promoted into ``out_scale`` in ``TrtllmAttention.forward`` for NVFP4 path
        "skip_mla_rope_generation",  # handled in ``TrtllmAttention.forward`` for the test-only MLA path
        "timestep",  # consumed by sparse prediction before FMHA dispatch
        "sparse_attn_phase",  # host-resolved sparse phase, consumed before FMHA dispatch
    }
)

# ``thop.attention`` kwargs hard-wired to a literal at the call site (no
# rich object owns them). Sync test enforces both the kwarg name and the
# literal value.
_THOP_LITERALS: dict = {}


class FallbackFmha(Fmha):
    """Fallback FMHA implementation using the fused TRT-LLM thop attention op."""

    supports_skip_correction = True
    supports_workspace_reclamation = True

    # Head sizes with NO fused context FMHA kernel, per SM version. This is a
    # blocklist of proven-absent combinations, not a support matrix: sourced
    # from the runtime fallback warning "Fall back to unfused MHA ...
    # headSize = 64 ... in sm_103" and the silent-wrong-answer incident it
    # caused. Extend it as further combinations are proven.
    CONTEXT_FMHA_ABSENT_HEAD_DIMS: ClassVar[dict[int, tuple[int, ...]]] = {
        103: (64,),
    }

    # Exemptions to the blocklist above: KV cache dtypes whose fused context
    # FMHA kernel a full build carries for an otherwise-blocked (sm,
    # head_dim). FP8 KV on sm103/hd64 is proven by hand (no unfused-MHA
    # fallback at boot or under sustained load, deterministic greedy replays,
    # correct long-context recall through chunked prefill and block reuse);
    # the 16-bit KV entries are proven by ``test_context_fmha_kernel_presence``
    # (head-size-64 kernels present for matched 16-bit Q/KV across the SM100
    # family) and exercised by the SM103 L0 suites that run paged-context
    # attention with a BF16 KV cache. The NVFP4 KV entry is proven by the same
    # test (head-size-64 E4M3-Q / E2M1-KV kernels present across the SM100
    # family; the SM103 trtllm-gen table carries them with E4M3 and BF16
    # output). Every entry is an assertion about a
    # kernel set the running build may not contain (a build whose
    # ``--cuda_architectures`` omits the SM carries none of these), so
    # ``validate_metadata`` confirms each against the native kernel lookup
    # rather than trusting it: on a build without the kernel the combination
    # stays refused. Unlisted dtypes stay refused (fail closed).
    CONTEXT_FMHA_PRESENT_KV_DTYPES: ClassVar[dict[tuple[int, int], tuple[DataType, ...]]] = {
        (103, 64): (DataType.FP8, DataType.NVFP4, DataType.BF16, DataType.HALF),
    }

    @classmethod
    def validate_metadata(cls, metadata: "TrtllmAttentionMetadata") -> None:
        """Refuse paged-context FMHA when no kernel exists for this config.

        When ``use_paged_context_fmha`` is enabled but the fused context FMHA
        kernel is absent for this SM/head-size combination, attentionOp.cpp
        falls back to unfused MHA whose context path builds K/V from the
        current chunk only: the cached prefix is DROPPED from attention and
        then OVERWRITTEN by the chunk's write-back. Any request with a
        non-zero cached length -- block reuse, partial reuse, chunked
        prefill, speculative draft tokens -- returns a plausible wrong answer
        with no error. That fallback happens inside the C++ op, invisible to
        FMHA library selection, so the refusal must happen here at metadata
        construction, where the features are enabled.

        ``CONTEXT_FMHA_PRESENT_KV_DTYPES`` exempts combinations whose kernel
        was proven present by hand. That is an assertion about a kernel set
        that can change under it, so every exemption is confirmed against the
        native kernel lookup before it is honoured: a dropped kernel fails
        here instead of in flight. A build whose bindings predate the lookup
        cannot be checked, so its exemptions are not honoured (fail closed).
        """
        if not metadata.use_paged_context_fmha:
            return
        manager = metadata.kv_cache_manager
        head_dim = getattr(manager, "head_dim", None) if manager else None
        if head_dim is None:
            return
        head_dims = head_dim if isinstance(head_dim, list) else [head_dim]
        sm = get_sm_version()
        absent = [dim for dim in head_dims if dim in cls.CONTEXT_FMHA_ABSENT_HEAD_DIMS.get(sm, ())]
        kv_dtype = getattr(manager, "dtype", None)
        tokens_per_block = getattr(manager, "tokens_per_block", None)
        if (
            absent
            and kv_dtype is not None
            and tokens_per_block is not None
            and all(
                kv_dtype in cls.CONTEXT_FMHA_PRESENT_KV_DTYPES.get((sm, dim), ()) for dim in absent
            )
        ):
            kernel_exists = getattr(thop, "fused_context_fmha_kernel_exists", None)
            if kernel_exists is None:
                raise RuntimeError(
                    f"CONTEXT_FMHA_PRESENT_KV_DTYPES exempts head_dim {absent} "
                    f"on SM {sm} for KV cache dtype {kv_dtype}, but this "
                    f"build's bindings predate the "
                    f"fused_context_fmha_kernel_exists lookup, so the "
                    f"exemption cannot be verified against the kernels the "
                    f"build contains. An unverified exemption risks a silent "
                    f"fall back to unfused MHA that corrupts the cached "
                    f"prefix, so the combination stays refused. Rebuild the "
                    f"bindings, or use the FlashInfer attention backend."
                )
            # Probe with the output precision the kernel table pairs with
            # this KV precision (matched 16-bit output; FP8 output for the
            # FP8/NVFP4 KV kernels), mirroring the binding's probe
            # convention in test_context_fmha_kernel_presence.py.
            if kv_dtype in (DataType.FP8, DataType.NVFP4):
                probe_output_dtype = DataType.FP8
            else:
                probe_output_dtype = kv_dtype
            for dim in absent:
                if kernel_exists(
                    head_size=dim,
                    kv_cache_dtype=kv_dtype,
                    tokens_per_block=tokens_per_block,
                    output_dtype=probe_output_dtype,
                ):
                    continue
                raise RuntimeError(
                    f"CONTEXT_FMHA_PRESENT_KV_DTYPES exempts head_dim {dim} "
                    f"on SM {sm} for KV cache dtype {kv_dtype}, but this "
                    f"build contains no fused context FMHA kernel for that "
                    f"combination at {tokens_per_block} tokens per block. The "
                    f"exemption table asserts a kernel the kernel set does "
                    f"not have, so paged-context FMHA would fall back to "
                    f"unfused MHA and silently corrupt the cached prefix. "
                    f"Drop the ({sm}, {dim}) entry from "
                    f"FallbackFmha.CONTEXT_FMHA_PRESENT_KV_DTYPES "
                    f"once the kernel is gone, or build with it present. A "
                    f"build whose --cuda_architectures does not name SM {sm} "
                    f"carries no kernels for SM {sm} at all and lands here "
                    f"too; check the architecture list first."
                )
            logger.info(
                f"Paged-context FMHA enabled for head_dim {absent} on SM "
                f"{sm}: the fused context FMHA kernel is proven present for "
                f"KV cache dtype {kv_dtype}."
            )
            return
        if absent:
            features = [
                name
                for name in ("chunked_prefill", "cache_reuse", "has_speculative_draft_tokens")
                if getattr(metadata.runtime_features, name, False)
            ]
            raise RuntimeError(
                f"The TRTLLM attention backend has no fused context FMHA "
                f"kernel for head_dim {absent} on SM {sm}, but "
                f"{'/'.join(features)} requires attending to cached KV during "
                f"the context phase (use_paged_context_fmha). The unfused "
                f"fallback silently drops and corrupts the cached prefix, "
                f"producing plausible wrong answers. Use the FlashInfer "
                f"attention backend, or disable KV block reuse, chunked "
                f"prefill and speculative decoding for this model."
            )

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        sparse_algorithm = getattr(attn.sparse_params, "algorithm", None)
        if sparse_algorithm in ("deepseek_v4", "dsa"):
            if getattr(attn, "kv_cache_dtype", None) == "fp8_ds_mla":
                return False
            if get_sm_version() in (120, 121):
                return False
        return True

    def _is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        del k, v, phase
        # A verify group may straddle a page boundary onto two CP ranks, so
        # its KV ownership is per-token. The fused thop path cannot express
        # that: its spec-dec mask and the per-sequence helix_is_inactive_rank
        # gate both assume the new KV entries are the trailing slots of one
        # rank's kv_len. Reject rather than run it silently wrong; being last
        # in the library list, this makes dispatch raise.
        if (
            metadata.helix_position_offsets is not None
            and metadata._helix_spec_tokens_valid
            and metadata.num_generations > 0
            # Count generation tokens only: in a mixed batch ``q`` also holds
            # the context tokens, which would otherwise trip this on a batch
            # that has exactly one query token per generation sequence.
            and q.shape[0] - metadata.num_ctx_tokens > metadata.num_generations
        ):
            return False
        if q is not None and q.dtype == torch.float8_e4m3fn:
            return False
        if forward_args.attention_mask == CustomAttentionMask.CUSTOM:
            return False
        if not forward_args.update_kv_cache and not metadata.is_cross:
            return False
        return True

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
        # (with ``forward_args.sparse_runtime_params`` for sparse inputs),
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
            max_num_sequences=metadata.max_num_sequences,
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
            force_prepare_spec_dec_tree_mask=metadata.force_prepare_spec_dec_tree_mask,
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
            dsv4_inv_rope_cos_sin_cache=forward_args.dsv4_inv_rope_cos_sin_cache,
            enable_dsv4_epilogue_fusion=forward_args.enable_dsv4_epilogue_fusion,
            kv_norm_weight=forward_args.kv_norm_weight,
            kv_norm_eps=forward_args.kv_norm_eps,
            sage_attn_num_elts_per_blk_q=forward_args.sage_attn_num_elts_per_blk_q,
            sage_attn_num_elts_per_blk_k=forward_args.sage_attn_num_elts_per_blk_k,
            sage_attn_num_elts_per_blk_v=forward_args.sage_attn_num_elts_per_blk_v,
            sage_attn_qk_int8=forward_args.sage_attn_qk_int8,
            is_fused_qkv=forward_args.is_fused_qkv,
            update_kv_cache=forward_args.update_kv_cache,
            cross_kv=forward_args.cross_kv,
            relative_attention_bias=forward_args.relative_attention_bias,
            relative_attention_max_distance=forward_args.relative_attention_max_distance,
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
            skip_correction_threshold=attn.skip_correction_threshold,
            uses_spcompress=attn.uses_spcompress,
            # --- Sparse runtime parameters ---
            sparse_kv_indices=forward_args.sparse_runtime_params.sparse_kv_indices,
            sparse_kv_offsets=forward_args.sparse_runtime_params.sparse_kv_offsets,
            sparse_attn_indices=forward_args.sparse_runtime_params.sparse_attn_indices,
            sparse_attn_offsets=forward_args.sparse_runtime_params.sparse_attn_offsets,
            sparse_attn_indices_block_size=(
                forward_args.sparse_runtime_params.sparse_attn_indices_block_size
            ),
            sparse_attn_kv_lens=forward_args.sparse_runtime_params.sparse_attn_kv_lens,
            aux_kv_cache_pool_ptr=forward_args.sparse_runtime_params.aux_kv_cache_pool_ptr,
            skip_softmax_threshold_scale_factor_prefill=(
                forward_args.sparse_runtime_params.threshold_scale_factor_prefill
            ),
            skip_softmax_threshold_scale_factor_decode=(
                forward_args.sparse_runtime_params.threshold_scale_factor_decode
            ),
        )
