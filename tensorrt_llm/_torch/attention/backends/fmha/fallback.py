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

from typing import TYPE_CHECKING, Dict, Optional, cast

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal import thop

from ..sparse.params import SparseRuntimeParams
from .interface import FmhaPhase, ensure_fmha_scheduler_counter
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


# ``TrtllmAttention`` caches a scheduler counter per layer, but the entry points below
# are ``@staticmethod``s with no instance to hang one off, so they share a per-device
# buffer. Keeping it cached rather than allocating per call also keeps the pointer
# stable across CUDA graph replays.
_SCHEDULER_COUNTERS: Dict[torch.device, torch.Tensor] = {}


def _cached_scheduler_counter(
    device: torch.device, num_heads: int, max_num_sequences: int
) -> torch.Tensor:
    counter = ensure_fmha_scheduler_counter(
        _SCHEDULER_COUNTERS.get(device), device, num_heads, max_num_sequences
    )
    _SCHEDULER_COUNTERS[device] = counter
    return counter


class FallbackFmha(PhasedFmha):
    """Fallback FMHA implementation over the phased TRT-LLM thop ops."""

    REQUIRES_PAGED_KV = False
    supports_skip_correction = True

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
        arguments = locals().copy()
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

        # The generation kernels dereference the scheduler counter unconditionally in
        # multi-block mode; this entry point has no other source for it.
        scheduler_counter = (
            _cached_scheduler_counter(qkv_or_q.device, num_heads, max_num_requests)
            if num_generations > 0
            else None
        )

        params = FmhaParams._from_arguments(
            arguments,
            layer_idx=0,
            beam_width=1,
            fwd=AttentionForwardArgs(
                fmha_scheduler_counter=scheduler_counter,
                chunked_prefill_buffer_batch_size=max_num_requests,
                sparse_runtime_params=SparseRuntimeParams(sparse_attn_indices_block_size=1),
            ),
            rotary_embedding_dim=rope_dim,
            rotary_embedding_base=10000.0,
            rotary_embedding_scale_type=0,
            rotary_embedding_scale=1.0,
            rotary_embedding_short_mscale=1.0,
            rotary_embedding_long_mscale=1.0,
            rotary_embedding_max_positions=max_context_length,
            rotary_embedding_original_max_positions=max_context_length,
            paged_context_fmha=True,
            is_mla_enable=False,
            is_spec_dec_tree=False,
            force_prepare_spec_dec_tree_mask=False,
            predicted_tokens_per_seq=1,
            num_sparse_topk=0,
            max_num_sequences=max_num_requests,
            cyclic_attention_window_size=attention_window_size,
            max_attention_window_size=attention_window_size,
        )
        thop_params = params.to_thop_params()

        # No layer object to hang an op off, so this entry point builds one per call --
        # the behaviour every caller had before ops became reusable.
        op = thop.AttentionOp()

        workspace_size = op.get_attention_workspace_size(
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
            thop_params.sequence_length = sequence_length[:num_contexts]
            thop_params.context_lengths = context_lengths[:num_contexts]
            thop_params.input_seq_length = int(host_context_lengths[:num_contexts].max())
            thop_params.max_past_kv_length = int(host_past_key_value_lengths[:num_contexts].max())
            thop_params.seq_offset = 0
            thop_params.num_seqs = num_contexts
            thop_params.num_requests = num_contexts
            thop_params.token_offset = 0
            thop_params.num_tokens = num_ctx_tokens
            thop_params.total_kv_len = int(host_total_kv_lens[0])
            op.run_context(thop_params)

        if num_generations > 0:
            thop_params.qkv_or_q = qkv_or_q[num_ctx_tokens : num_ctx_tokens + num_gen_tokens]
            thop_params.output = output[num_ctx_tokens : num_ctx_tokens + num_gen_tokens]
            thop_params.sequence_length = sequence_length[num_contexts:]
            thop_params.context_lengths = context_lengths[num_contexts:]
            thop_params.input_seq_length = num_gen_tokens // num_generations
            thop_params.max_past_kv_length = int(host_past_key_value_lengths[num_contexts:].max())
            thop_params.seq_offset = num_contexts
            thop_params.num_seqs = num_generations
            thop_params.num_requests = num_generations
            thop_params.token_offset = num_ctx_tokens
            thop_params.num_tokens = num_gen_tokens
            thop_params.total_kv_len = int(host_total_kv_lens[1])
            op.run_generation(thop_params)

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        sparse_algorithm = getattr(attn.sparse_params, "algorithm", None)
        if sparse_algorithm in ("deepseek_v4", "dsa"):
            if getattr(attn, "kv_cache_dtype", None) == "fp8_ds_mla":
                return False
            if get_sm_version() in (120, 121):
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
        del q, k, v, phase
        return forward_args.attention_mask != CustomAttentionMask.CUSTOM and (
            forward_args.update_kv_cache or metadata.is_cross
        )

    @staticmethod
    def attention(
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
        host_request_types: torch.Tensor,
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
        spec_decoding_position_offsets_for_cpp: Optional[torch.Tensor],
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
    ) -> None:
        """Drop-in replacement for the removed monolithic ``thop.attention``.

        Builds a single native FMHA parameter holder and dispatches to the phased
        ``AttentionOp.run_context`` / ``run_generation`` / ``run_mla_generation``
        ops, preserving the old call signature so AutoDeploy call sites only need to
        change ``thop.attention(`` -> ``FallbackFmha.attention(``.
        """
        arguments = locals().copy()
        del host_request_types, update_kv_cache, skip_softmax_stat

        if workspace is None:
            raise RuntimeError("FallbackFmha.attention requires workspace.")
        if output is None:
            raise RuntimeError("FallbackFmha.attention requires output.")

        num_tokens = q.size(0)
        if attention_input_type is None:
            attention_input_type = AttentionInputType.mixed
        is_gen_only = attention_input_type == AttentionInputType.generation_only
        is_ctx_only = attention_input_type == AttentionInputType.context_only
        if is_gen_only:
            num_contexts = 0
            num_ctx_tokens = 0
        total_seqs = host_context_lengths.size(0)
        num_generations = 0 if is_ctx_only else total_seqs - num_contexts
        num_gen_tokens = num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid FMHA token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}."
            )

        # Generation requires a scheduler counter: it backs both the MMHA/XQA
        # multi-block counter and the MLA FMHA tile counter, and neither consumer
        # checks for null.
        if num_generations > 0 and fmha_scheduler_counter is None:
            fmha_scheduler_counter = _cached_scheduler_counter(
                q.device, num_heads, max_num_sequences or max_num_requests
            )

        params = FmhaParams._from_arguments(
            arguments,
            qkv_or_q=q,
            layer_idx=local_layer_idx,
            fwd=AttentionForwardArgs(
                fmha_scheduler_counter=fmha_scheduler_counter,
                chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size or 1,
                sparse_runtime_params=SparseRuntimeParams(
                    threshold_scale_factor_prefill=skip_softmax_threshold_scale_factor_prefill
                    or 0.0,
                    threshold_scale_factor_decode=skip_softmax_threshold_scale_factor_decode or 0.0,
                ),
            ),
            tokens_per_block=tokens_per_block or 0,
            rotary_embedding_dim=rope_dim,
            rotary_embedding_base=rope_base,
            rotary_embedding_scale_type=rope_scale_type,
            rotary_embedding_scale=rope_scale,
            rotary_embedding_short_mscale=rope_short_m_scale,
            rotary_embedding_long_mscale=rope_long_m_scale,
            rotary_embedding_max_positions=rope_max_positions,
            rotary_embedding_original_max_positions=rope_original_max_positions,
            paged_context_fmha=use_paged_context_fmha,
            kv_lora_rank=kv_lora_rank or 0,
            qk_nope_head_dim=qk_nope_head_dim or 0,
            qk_rope_head_dim=qk_rope_head_dim or 0,
            num_sparse_topk=num_sparse_topk or 0,
            max_num_sequences=max_num_sequences or max_num_requests,
            cyclic_attention_window_size=attention_window_size,
            max_attention_window_size=(
                attention_window_size
                if beam_width == 1 or cache_indirection is None
                else cache_indirection.size(2)
            ),
        )
        tp = params.to_thop_params()
        op = thop.AttentionOp()

        max_blocks_per_sequence = (
            kv_cache_block_offsets.size(-1) if kv_cache_block_offsets is not None else 0
        )
        workspace_size = op.get_attention_workspace_size(
            tp,
            num_tokens,
            attention_window_size,
            num_gen_tokens,
            max_blocks_per_sequence,
            int(host_total_kv_lens[0]),
        )
        if workspace.numel() < workspace_size:
            workspace.resize_(workspace_size)

        if num_contexts > 0 and not is_gen_only:
            # Context phase. The context-MLA path is handled inside run_context, so
            # both MLA and non-MLA go through run_context.
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
                max_context_q_len = override
                max_past_kv_len = override
            tp.qkv_or_q = q[:num_ctx_tokens]
            tp.k = k[:num_ctx_tokens] if k is not None else None
            tp.v = v[:num_ctx_tokens] if v is not None else None
            tp.output = output[:num_ctx_tokens]
            tp.sequence_length = sequence_length
            tp.context_lengths = context_lengths
            tp.input_seq_length = max_context_q_len
            tp.max_past_kv_length = max_past_kv_len
            tp.seq_offset = 0
            tp.num_seqs = num_contexts
            tp.num_requests = num_contexts
            tp.token_offset = 0
            tp.num_tokens = num_ctx_tokens
            tp.total_kv_len = int(host_total_kv_lens[0])
            op.run_context(tp)

        if num_generations > 0 and not is_ctx_only:
            seq_offset = num_contexts
            tp.qkv_or_q = q[num_ctx_tokens:]
            tp.k = k[num_ctx_tokens:] if k is not None else None
            tp.v = v[num_ctx_tokens:] if v is not None else None
            tp.output = output[num_ctx_tokens:]
            tp.sequence_length = sequence_length[seq_offset:]
            tp.context_lengths = context_lengths[seq_offset:]
            tp.input_seq_length = num_gen_tokens // num_generations
            tp.max_past_kv_length = int(host_past_key_value_lengths[seq_offset:].max())
            tp.seq_offset = seq_offset
            tp.num_seqs = num_generations
            tp.num_requests = num_generations // beam_width
            # The tensors above are phase-local; token_offset only indexes the whole-batch
            # FP4 scaling-factor output.
            tp.token_offset = num_ctx_tokens
            tp.num_tokens = num_gen_tokens
            tp.total_kv_len = int(host_total_kv_lens[1])
            if is_mla_enable:
                op.run_mla_generation(tp)
            else:
                op.run_generation(tp)

    def _to_thop_params(self, params: FmhaParams) -> "thop.FmhaParams":
        """Validate and lower one phase's Python parameters."""
        if params.fwd is None:
            raise RuntimeError("FallbackFmha requires forward args.")
        if params.output is None:
            raise RuntimeError("FallbackFmha requires output.")
        if params.qkv_or_q is None:
            raise RuntimeError("FallbackFmha requires qkv_or_q.")
        if params.workspace is None:
            raise RuntimeError("FallbackFmha requires workspace.")
        return params.to_thop_params()

    def prepare_workspace(
        self,
        params: FmhaParams,
        metadata: "TrtllmAttentionMetadata",
    ) -> None:
        tp = self._to_thop_params(params)
        q = cast(torch.Tensor, params.qkv_or_q)
        workspace = cast(torch.Tensor, params.workspace)
        fwd = cast(AttentionForwardArgs, params.fwd)

        num_tokens = q.size(0)
        is_gen_only = fwd.attention_input_type == AttentionInputType.generation_only
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - metadata.num_ctx_tokens
        kv_cache_block_offsets = params.kv_cache_block_offsets
        use_kv_cache = kv_cache_block_offsets is not None
        max_blocks_per_sequence = kv_cache_block_offsets.size(-1) if use_kv_cache else 0
        max_attention_window_size = (
            params.cyclic_attention_window_size
            if metadata.effective_beam_width == 1
            else params.max_attention_window_size
        )
        workspace_size = self.attn.attention_op(params).get_attention_workspace_size(
            tp,
            num_tokens,
            max_attention_window_size,
            num_gen_tokens,
            max_blocks_per_sequence,
            int(metadata.host_total_kv_lens[0]),
        )
        if workspace.numel() < workspace_size:
            workspace.resize_(workspace_size)

    def run_context(self, params: FmhaParams) -> None:
        self.attn.attention_op(params).run_context(self._to_thop_params(params))

    def run_mla_context(self, params: FmhaParams) -> None:
        self.attn.attention_op(params).run_context(self._to_thop_params(params))

    def run_generation(self, params: FmhaParams) -> None:
        self.attn.attention_op(params).run_generation(self._to_thop_params(params))

    def run_mla_generation(self, params: FmhaParams) -> None:
        self.attn.attention_op(params).run_mla_generation(self._to_thop_params(params))
