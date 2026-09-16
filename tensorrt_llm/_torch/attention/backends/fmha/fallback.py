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

import atexit
from dataclasses import fields
from threading import Thread, current_thread
from typing import TYPE_CHECKING, Mapping, Optional, cast

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
)
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType

from ..sparse.params import SparseRuntimeParams
from .interface import FmhaPhase, StaticAttentionConfig
from .phased import FmhaParams, PhasedFmha
from .utils import get_multi_ctas_kv_counter

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


class _AttentionOpCache:
    """Reuse initialized runners, without retaining any per-call tensors."""

    def __init__(self) -> None:
        self._ops: dict[tuple[Thread, torch.device, StaticAttentionConfig], thop.AttentionOp] = {}

    def get(self, config: StaticAttentionConfig, device: torch.device) -> "thop.AttentionOp":
        # Native runners and cuBLAS state must not be shared by concurrent host threads.
        # Do not key by stream: CUDA graph capture must reuse the op initialized during
        # warmup on another stream. Execution passes the current stream and scratch anew.
        key = (current_thread(), device, config)
        op = self._ops.get(key)
        if op is None:
            with torch.cuda.device(device):
                op = thop.AttentionOp(config.to_thop_config())
            self._ops[key] = op
        return op

    def clear(self) -> None:
        self._ops.clear()


# The stateless compatibility entry point has no layer owner. Keep its runners alive
# across calls/captures, like the old native cache, and release them before CUDA teardown.
_compat_attention_ops = _AttentionOpCache()
atexit.register(_compat_attention_ops.clear)


def _compat_attention_op(params: FmhaParams) -> "thop.AttentionOp":
    config = StaticAttentionConfig.from_params(params)
    return _compat_attention_ops.get(config, cast(torch.Tensor, params.qkv_or_q).device)


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


def _set_context_workspace_shape(
    params: FmhaParams,
    *,
    num_contexts: int,
    num_ctx_tokens: int,
    host_context_lengths: torch.Tensor,
    host_past_key_value_lengths: torch.Tensor,
    host_total_kv_lens: torch.Tensor,
    max_context_q_len_override: Optional[int],
) -> None:
    """Set the active context extents consumed by native workspace sizing."""
    if num_contexts <= 0 or num_ctx_tokens <= 0:
        params.num_seqs = 0
        params.num_requests = 0
        params.num_tokens = 0
        params.input_seq_length = 0
        params.max_past_kv_length = 0
        params.total_kv_len = 0
        return

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

    params.num_seqs = num_contexts
    params.num_requests = num_contexts
    params.num_tokens = num_ctx_tokens
    params.input_seq_length = max_context_q_len
    params.max_past_kv_length = max_past_kv_len
    params.total_kv_len = int(host_total_kv_lens[0])


class FallbackFmha(PhasedFmha):
    """Fallback FMHA implementation over the phased TRT-LLM thop ops."""

    REQUIRES_PAGED_KV = False
    supports_skip_correction = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._multi_ctas_kv_counter: Optional[torch.Tensor] = None
        # Construct lazily: the layer can be initialized on meta before CUDA is ready.
        self._attention_ops = _AttentionOpCache()

    def attention_op(self, params: FmhaParams) -> "thop.AttentionOp":
        config = StaticAttentionConfig.from_params(
            params, skip_correction_threshold=self.attn.skip_correction_threshold
        )
        return self._attention_ops.get(config, cast(torch.Tensor, params.qkv_or_q).device)

    def release(self) -> None:
        self._attention_ops.clear()
        self._multi_ctas_kv_counter = None

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
        if q is not None and q.dtype == torch.float8_e4m3fn:
            return False
        if forward_args.attention_mask == CustomAttentionMask.CUSTOM:
            return False
        if not forward_args.update_kv_cache and not metadata.is_cross:
            return False
        return True

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
        """Shared MHA/MLA replacement for the removed monolithic ``thop.attention``.

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
            # Q is decode-only, but host metadata / KV blocks can still include
            # a prefill prefix. Preserve num_contexts as the sequence offset.
            num_ctx_tokens = 0
        total_seqs = host_context_lengths.size(0)
        num_generations = 0 if is_ctx_only else total_seqs - num_contexts
        num_gen_tokens = num_tokens - num_ctx_tokens
        if num_gen_tokens < 0:
            raise RuntimeError(
                f"Invalid FMHA token counts: num_tokens={num_tokens}, "
                f"num_ctx_tokens={num_ctx_tokens}."
            )

        # Generation requires an FMHA-owned multi-CTA scratch buffer in addition to
        # MLA's caller-owned scheduler counter.
        multi_ctas_kv_counter = None
        if num_generations > 0:
            multi_ctas_kv_counter = get_multi_ctas_kv_counter(
                None, q.device, num_heads, max_num_sequences or max_num_requests
            )

        params = FmhaParams._from_arguments(
            arguments,
            qkv_or_q=q,
            layer_idx=local_layer_idx,
            multi_ctas_kv_counter=multi_ctas_kv_counter,
            fwd=_legacy_forward_args(
                arguments,
                chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size or 1,
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
        _set_context_workspace_shape(
            params,
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            host_context_lengths=host_context_lengths,
            host_past_key_value_lengths=host_past_key_value_lengths,
            host_total_kv_lens=host_total_kv_lens,
            max_context_q_len_override=max_context_q_len_override,
        )
        tp = params.to_thop_params()
        op = _compat_attention_op(params)

        max_blocks_per_sequence = (
            kv_cache_block_offsets.size(-1) if kv_cache_block_offsets is not None else 0
        )
        workspace_size = op.get_attention_workspace_size(
            tp,
            num_tokens,
            attention_window_size,
            num_gen_tokens,
            max_blocks_per_sequence,
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
            if k is not None:
                tp.k = k[:num_ctx_tokens]
            if v is not None:
                tp.v = v[:num_ctx_tokens]
            tp.output = output[:num_ctx_tokens]
            tp.sequence_length = sequence_length[:num_contexts]
            tp.context_lengths = context_lengths[:num_contexts]
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
            if k is not None:
                tp.k = k[num_ctx_tokens:]
            if v is not None:
                tp.v = v[num_ctx_tokens:]
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
        params.multi_ctas_kv_counter = self._multi_ctas_kv_counter
        return params.to_thop_params()

    # Keep nanobind calls eager, including when CombinedFmha delegates individual phases.
    @torch.compiler.disable
    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        if metadata.num_generations > 0:
            self._multi_ctas_kv_counter = get_multi_ctas_kv_counter(
                self._multi_ctas_kv_counter,
                q.device,
                self.attn.num_heads,
                metadata.max_num_sequences or metadata.max_num_requests,
            )
        # Native sizing needs the full configuration, but other FMHA libraries
        # should not need a native parameter carrier just to size their workspace.
        params = self._build_params(q, k, v, metadata, forward_args, workspace)
        num_tokens = q.size(0)
        is_gen_only = forward_args.attention_input_type == AttentionInputType.generation_only
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - metadata.num_ctx_tokens
        _set_context_workspace_shape(
            params,
            num_contexts=0 if is_gen_only else metadata.num_contexts,
            num_ctx_tokens=0 if is_gen_only else metadata.num_ctx_tokens,
            host_context_lengths=cast(torch.Tensor, params.host_context_lengths),
            host_past_key_value_lengths=cast(torch.Tensor, params.host_past_key_value_lengths),
            host_total_kv_lens=metadata.host_total_kv_lens,
            max_context_q_len_override=params.max_context_q_len_override,
        )
        tp = self._to_thop_params(params)

        kv_cache_block_offsets = params.kv_cache_block_offsets
        use_kv_cache = kv_cache_block_offsets is not None
        max_blocks_per_sequence = kv_cache_block_offsets.size(-1) if use_kv_cache else 0
        max_attention_window_size = (
            params.cyclic_attention_window_size
            if metadata.effective_beam_width == 1
            else params.max_attention_window_size
        )
        workspace_size = self.attention_op(params).get_attention_workspace_size(
            tp,
            num_tokens,
            max_attention_window_size,
            num_gen_tokens,
            max_blocks_per_sequence,
        )
        if workspace.numel() < workspace_size:
            workspace.resize_(workspace_size)

    @torch.compiler.disable
    def run_context(self, params: FmhaParams) -> None:
        self.attention_op(params).run_context(self._to_thop_params(params))

    @torch.compiler.disable
    def run_mla_context(self, params: FmhaParams) -> None:
        self.attention_op(params).run_context(self._to_thop_params(params))

    @torch.compiler.disable
    def run_generation(self, params: FmhaParams) -> None:
        self.attention_op(params).run_generation(self._to_thop_params(params))

    @torch.compiler.disable
    def run_mla_generation(self, params: FmhaParams) -> None:
        self.attention_op(params).run_mla_generation(self._to_thop_params(params))
