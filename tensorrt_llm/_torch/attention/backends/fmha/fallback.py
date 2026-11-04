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

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Mapping, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm.functional import AttentionMaskType

from ..sparse.params import SparseRuntimeParams
from .flash_mla import FlashMlaFmha
from .fmha_v2 import FmhaV2Fmha
from .mmha import MmhaFmha
from .native import NativeFmha
from .phased import FmhaParams
from .trtllm_gen import TrtllmGenFmha
from .unfused import UnfusedFmha
from .xqa import XqaFmha

_FORWARD_ARG_NAMES = frozenset(field.name for field in fields(AttentionForwardArgs))
_SPARSE_ARG_NAMES = frozenset(field.name for field in fields(SparseRuntimeParams))


def _legacy_forward_args(
    arguments: Mapping[str, object], **overrides: object
) -> AttentionForwardArgs:
    """Translate flat compatibility arguments into the shared nested carriers."""
    sparse = {
        name: arguments[name] for name in _SPARSE_ARG_NAMES if arguments.get(name) is not None
    }
    sparse.setdefault("sparse_attn_indices_block_size", 1)
    sparse["threshold_scale_factor_prefill"] = (
        arguments.get("skip_softmax_threshold_scale_factor_prefill") or 0.0
    )
    sparse["threshold_scale_factor_decode"] = (
        arguments.get("skip_softmax_threshold_scale_factor_decode") or 0.0
    )
    values = {
        name: arguments[name] for name in _FORWARD_ARG_NAMES if arguments.get(name) is not None
    }
    mask_type = arguments["mask_type"]
    if mask_type == AttentionMaskType.causal:
        values["attention_mask"] = PredefinedAttentionMask.CAUSAL
    elif mask_type == AttentionMaskType.padding:
        values["attention_mask"] = PredefinedAttentionMask.FULL
    else:
        raise ValueError(f"Unsupported legacy attention mask type: {mask_type}")
    values["sparse_runtime_params"] = SparseRuntimeParams(**sparse)
    values.update(overrides)
    return AttentionForwardArgs(**values)


# The native op's library priorities. They agree phase by phase with the registry order,
# which serves both variants: fmha_v2 serves no MHA generation and XQA no context phase.
_MHA_LIBRARIES: tuple[type[NativeFmha], ...] = (
    TrtllmGenFmha,
    FmhaV2Fmha,
    XqaFmha,
    MmhaFmha,
    UnfusedFmha,
)
_MLA_LIBRARIES: tuple[type[NativeFmha], ...] = (
    TrtllmGenFmha,
    FlashMlaFmha,
    XqaFmha,
    FmhaV2Fmha,
)


def _select_native_library(params: FmhaParams, is_gen: bool) -> type[NativeFmha]:
    """The first native library that serves this phase call."""
    libraries = _MLA_LIBRARIES if params.attn.is_mla_enable else _MHA_LIBRARIES
    for library in libraries:
        serves_phase = library.SERVES_GENERATION if is_gen else library.SERVES_CONTEXT
        if serves_phase and library.native_is_supported(params, is_gen):
            return library
    raise RuntimeError(
        f"No native FMHA library supports this {'generation' if is_gen else 'context'} call."
    )


class FallbackFmha(NativeFmha):
    """The native libraries as one library: each phase call runs the first that serves it.

    The order is the native op's own priority for the layer's attention variant.
    """

    SERVES_CONTEXT = True
    SERVES_GENERATION = True

    @staticmethod
    def native_is_supported(params: FmhaParams, is_gen: bool) -> bool:
        try:
            _select_native_library(params, is_gen)
        except RuntimeError:
            return False
        return True

    @staticmethod
    def native_run_context(params: FmhaParams) -> None:
        _select_native_library(params, False).native_run_context(params)

    @staticmethod
    def native_run_generation(params: FmhaParams) -> None:
        _select_native_library(params, True).native_run_generation(params)

    @classmethod
    def attention(
        cls,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        output: torch.Tensor,
        output_sf: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        sequence_length: torch.Tensor,
        host_past_key_value_lengths: torch.Tensor,
        host_total_kv_lens: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths: torch.Tensor,
        host_request_types: Optional[torch.Tensor],
        max_context_q_len_override: Optional[int],
        kv_cache_block_offsets: Optional[torch.Tensor],
        host_kv_cache_pool_pointers: Optional[torch.Tensor],
        host_kv_cache_pool_mapping: Optional[torch.Tensor],
        cache_indirection: Optional[torch.Tensor],
        kv_scale_orig_quant: Optional[torch.Tensor],
        kv_scale_quant_orig: Optional[torch.Tensor],
        out_scale: Optional[torch.Tensor],
        rotary_inv_freq: Optional[torch.Tensor],
        rotary_cos_sin: Optional[torch.Tensor],
        latent_cache: Optional[torch.Tensor],
        q_pe: Optional[torch.Tensor],
        block_ids_per_seq: Optional[torch.Tensor],
        attention_sinks: Optional[torch.Tensor],
        is_fused_qkv: bool,
        update_kv_cache: bool,
        predicted_tokens_per_seq: int,
        local_layer_idx: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        tokens_per_block: Optional[int],
        max_num_requests: int,
        max_context_length: int,
        max_seq_len: int,
        attention_window_size: int,
        beam_width: int,
        mask_type: int,
        quant_mode: int,
        q_scaling: float,
        position_embedding_type: int,
        rope_dim: int,
        rope_base: float,
        rope_scale_type: int,
        rope_scale: float,
        rope_short_m_scale: float,
        rope_long_m_scale: float,
        rope_max_positions: int,
        rope_original_max_positions: int,
        use_paged_context_fmha: bool,
        attention_input_type: Optional[int],
        is_mla_enable: bool,
        chunked_prefill_buffer_batch_size: Optional[int],
        q_lora_rank: Optional[int],
        kv_lora_rank: Optional[int],
        qk_nope_head_dim: Optional[int],
        qk_rope_head_dim: Optional[int],
        v_head_dim: Optional[int],
        rope_append: Optional[bool],
        mrope_rotary_cos_sin: Optional[torch.Tensor],
        mrope_position_deltas: Optional[torch.Tensor],
        helix_position_offsets: Optional[torch.Tensor],
        helix_is_inactive_rank: Optional[torch.Tensor],
        attention_chunk_size: Optional[int],
        softmax_stats_tensor: Optional[torch.Tensor],
        is_spec_decoding_enabled: bool,
        use_spec_decoding: bool,
        is_spec_dec_tree: bool,
        spec_decoding_generation_lengths: Optional[torch.Tensor],
        spec_decoding_position_offsets: Optional[torch.Tensor],
        spec_decoding_packed_mask: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask_offset: Optional[torch.Tensor],
        spec_decoding_bl_tree_mask: Optional[torch.Tensor],
        spec_bl_tree_first_sparse_mask_offset_kv: Optional[torch.Tensor],
        sparse_kv_indices: Optional[torch.Tensor],
        sparse_kv_offsets: Optional[torch.Tensor],
        sparse_attn_indices: Optional[torch.Tensor],
        sparse_attn_offsets: Optional[torch.Tensor],
        sparse_attn_indices_block_size: int,
        num_sparse_topk: Optional[int] = None,
        sparse_attn_kv_lens: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor_prefill: Optional[float] = None,
        skip_softmax_threshold_scale_factor_decode: Optional[float] = None,
        skip_softmax_stat: Optional[torch.Tensor] = None,
        cu_q_seqlens: Optional[torch.Tensor] = None,
        cu_kv_seqlens: Optional[torch.Tensor] = None,
        fmha_scheduler_counter: Optional[torch.Tensor] = None,
        mla_bmm1_scale: Optional[torch.Tensor] = None,
        mla_bmm2_scale: Optional[torch.Tensor] = None,
        quant_q_buffer: Optional[torch.Tensor] = None,
        flash_mla_tile_scheduler_metadata: Optional[torch.Tensor] = None,
        flash_mla_num_splits: Optional[torch.Tensor] = None,
        sage_attn_num_elts_per_blk_q: int = 0,
        sage_attn_num_elts_per_blk_k: int = 0,
        sage_attn_num_elts_per_blk_v: int = 0,
        sage_attn_qk_int8: bool = False,
        sage_attn_smooth_k: bool = False,
        num_contexts: int = 0,
        num_ctx_tokens: int = 0,
        trtllm_gen_jit_warmup: bool = False,
        aux_kv_cache_pool_ptr: Optional[int] = None,
        is_cross: bool = False,
        cross_kv: Optional[torch.Tensor] = None,
        relative_attention_bias: Optional[torch.Tensor] = None,
        relative_attention_max_distance: int = 0,
        spec_decoding_target_max_draft_tokens: Optional[int] = None,
        quant_scale_qkv: Optional[torch.Tensor] = None,
        dsv4_inv_rope_cos_sin_cache: Optional[torch.Tensor] = None,
        enable_dsv4_epilogue_fusion: bool = False,
        force_prepare_spec_dec_tree_mask: bool = False,
        max_num_sequences: Optional[int] = None,
        kv_norm_weight: Optional[torch.Tensor] = None,
        kv_norm_eps: float = 1e-6,
        skip_correction_threshold: float = 0.0,
        uses_spcompress: Optional[bool] = None,
    ) -> None:
        """Shared MHA/MLA replacement for the removed monolithic ``thop.attention``.

        Builds native FMHA parameters and runs each phase on the first native
        library that serves it. Direct callers supply the flat attention
        arguments and workspace.
        """
        arguments = locals().copy()
        del host_request_types

        if workspace is None:
            raise RuntimeError("FallbackFmha.attention requires workspace.")
        if output is None:
            raise RuntimeError("FallbackFmha.attention requires output.")
        if not update_kv_cache and not is_cross:
            raise RuntimeError(
                "KV cache update cannot be disabled now (except for cross attention)."
            )

        num_tokens = q.size(0)
        if attention_input_type is None:
            attention_input_type = AttentionInputType.mixed
        is_gen_only = attention_input_type == AttentionInputType.generation_only
        is_ctx_only = attention_input_type == AttentionInputType.context_only
        if is_gen_only:
            # Q is decode-only, but host metadata / KV blocks can still include
            # a prefill prefix. Preserve num_contexts as the sequence offset.
            num_ctx_tokens = 0
        total_seqs = host_context_lengths.size(0)
        num_generations = 0 if is_ctx_only else total_seqs - num_contexts
        if (
            enable_dsv4_epilogue_fusion
            and num_contexts > 0
            and num_generations > 0
            and not is_gen_only
        ):
            raise ValueError("DSv4 fused epilogue requires separate context and generation calls.")
        num_gen_tokens = num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid FMHA token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}."
            )

        effective_beam_width = 1 if is_cross else beam_width
        max_attention_window_size = (
            attention_window_size
            if effective_beam_width == 1 or cache_indirection is None
            else cache_indirection.size(2)
        )
        has_context = num_contexts > 0 and not is_gen_only
        has_generation = num_generations > 0 and not is_ctx_only

        # The native reader walks `attn` and `meta` by attribute name, so the flat
        # arguments are presented under the names the attention layer and metadata use.
        attn = SimpleNamespace(
            layer_idx=local_layer_idx,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            is_mla_enable=is_mla_enable,
            v_head_dim=v_head_dim if v_head_dim is not None else head_size,
            rope_params=RopeParams(
                dim=rope_dim,
                theta=rope_base,
                scale_type=rope_scale_type,
                scale=rope_scale,
                short_m_scale=rope_short_m_scale,
                long_m_scale=rope_long_m_scale,
                max_positions=rope_max_positions,
                original_max_positions=rope_original_max_positions,
            ),
            q_scaling=q_scaling,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            skip_correction_threshold=skip_correction_threshold,
            uses_spcompress=bool(uses_spcompress),
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            rope_append=rope_append,
            rotary_inv_freq=rotary_inv_freq,
            rotary_cos_sin=rotary_cos_sin,
            position_embedding_type=position_embedding_type,
            skip_softmax_stat=skip_softmax_stat,
            quant_mode=quant_mode,
            attention_chunk_size=attention_chunk_size,
            get_local_layer_idx=lambda metadata: local_layer_idx,
        )
        meta = SimpleNamespace(
            max_num_requests=max_num_requests,
            max_num_sequences=max_num_sequences,
            cache_indirection=cache_indirection,
            effective_beam_width=effective_beam_width,
            max_seq_len=max_seq_len,
            max_context_length=max_context_length,
            max_context_q_len_override=max_context_q_len_override,
            is_spec_decoding_enabled=is_spec_decoding_enabled,
            is_spec_dec_tree=is_spec_dec_tree,
            force_prepare_spec_dec_tree_mask=force_prepare_spec_dec_tree_mask,
            max_total_draft_tokens=spec_decoding_target_max_draft_tokens,
            trtllm_gen_jit_warmup=trtllm_gen_jit_warmup,
            helix_position_offsets=helix_position_offsets,
            helix_is_inactive_rank=helix_is_inactive_rank,
            kv_cache_block_offsets=kv_cache_block_offsets,
            block_ids_per_seq=block_ids_per_seq,
            flash_mla_tile_scheduler_metadata=flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=flash_mla_num_splits,
            use_paged_context_fmha=use_paged_context_fmha,
            num_sparse_topk=num_sparse_topk or 0,
            host_kv_cache_pool_pointers=host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=host_kv_cache_pool_mapping,
            kv_lens_runtime=host_past_key_value_lengths,
            prompt_lens_cpu_runtime=host_context_lengths,
            prompt_lens_cuda_runtime=context_lengths,
            is_cross=is_cross,
        )
        fwd = _legacy_forward_args(
            arguments,
            chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size or 1,
        )

        def phase_params(token_lo: int, token_hi: int, seq_lo: int, seq_hi: int) -> FmhaParams:
            return FmhaParams(
                attn=attn,
                meta=meta,
                fwd=fwd,
                workspace=workspace,
                qkv_input=q[token_lo:token_hi],
                key_input=k[token_lo:token_hi] if k is not None else None,
                value_input=v[token_lo:token_hi] if v is not None else None,
                output=output if enable_dsv4_epilogue_fusion else output[token_lo:token_hi],
                sequence_lengths=sequence_length[seq_lo:seq_hi],
                context_lengths=context_lengths[seq_lo:seq_hi],
                host_total_kv_lens=host_total_kv_lens,
                max_attention_window_size=max_attention_window_size,
                cyclic_attention_window_size=attention_window_size,
                tokens_per_block=tokens_per_block or 0,
                kv_factor=1,
                batch_size=seq_hi - seq_lo,
                num_requests=seq_hi - seq_lo,
                seq_offset=seq_lo,
                token_offset=token_lo,
                num_tokens=token_hi - token_lo,
                use_spec_decoding=use_spec_decoding,
                spec_decoding_generation_lengths=spec_decoding_generation_lengths,
                spec_decoding_position_offsets=spec_decoding_position_offsets,
                spec_decoding_packed_mask=spec_decoding_packed_mask,
                spec_decoding_bl_tree_mask_offset=spec_decoding_bl_tree_mask_offset,
                spec_decoding_bl_tree_mask=spec_decoding_bl_tree_mask,
                spec_bl_tree_first_sparse_mask_offset_kv=spec_bl_tree_first_sparse_mask_offset_kv,
                is_cross=is_cross,
                local_layer_idx=local_layer_idx,
            )

        if has_context:
            max_context_q_len = int(host_context_lengths[:num_contexts].max())
            max_past_kv_len = int(host_past_key_value_lengths[:num_contexts].max())
            if max_context_q_len_override is not None:
                override = int(max_context_q_len_override)
                if override < max_context_q_len or override < max_past_kv_len:
                    raise ValueError(
                        f"max_context_q_len_override ({override}) must be >= the computed max "
                        f"context q length ({max_context_q_len}) and max past kv length "
                        f"({max_past_kv_len})."
                    )
            params = phase_params(0, num_ctx_tokens, 0, num_contexts)
            params.input_seq_length = max_context_q_len
            params.max_past_kv_length = max_past_kv_len
            # The context call reserves the generation phase's workspace as well.
            params.num_gen_tokens = num_gen_tokens if has_generation else 0
            cls.native_run_context(params)

        if has_generation:
            params = phase_params(num_ctx_tokens, num_tokens, num_contexts, total_seqs)
            params.input_seq_length = (
                num_gen_tokens // num_generations if num_generations > 0 else 1
            )
            params.max_past_kv_length = int(
                host_past_key_value_lengths[num_contexts:total_seqs].max()
            )
            params.num_requests = num_generations // effective_beam_width
            params.num_gen_tokens = num_gen_tokens
            cls.native_run_generation(params)
