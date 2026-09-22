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

from dataclasses import fields
from threading import Thread, current_thread
from typing import TYPE_CHECKING, ClassVar, Mapping, Optional

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
from . import interface
from .interface import FmhaPhase, StaticAttentionConfig, build_op_params
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
    params: FmhaParams, *, num_contexts: int, num_ctx_tokens: int
) -> None:
    """Point the phase counters at the context extent for native workspace sizing.

    Only the counts: the op reduces the host length arrays over `seq_offset` and
    `batch_size` itself, so the extents no longer cross the boundary.
    """
    active = num_contexts > 0 and num_ctx_tokens > 0
    params.seq_offset = 0
    params.batch_size = num_contexts if active else 0
    params.num_requests = num_contexts if active else 0
    params.num_tokens = num_ctx_tokens if active else 0


def _phase_query(params: FmhaParams) -> torch.Tensor:
    """The phase's query tensor, whichever of the two mutually exclusive fields holds it."""
    query = params.qkv_input if params.qkv_input is not None else params.query_input
    if query is None:
        raise RuntimeError("FallbackFmha requires qkv_input or query_input.")
    return query


class FallbackFmha(PhasedFmha):
    """Fallback FMHA implementation over the phased TRT-LLM thop ops."""

    REQUIRES_PAGED_KV = False
    NEEDS_BLOCK_EXTENT = False
    # The flat compatibility entry point is a classmethod with no layer instance, so its
    # runners hang off the class instead. Kept across calls and captures like the layer's
    # own cache, and cleared before CUDA teardown by the atexit hook below.
    _compat_attention_ops: ClassVar[_AttentionOpCache] = _AttentionOpCache()
    supports_skip_correction = True
    supports_workspace_reclamation = True

    def __init__(self, attn: "TrtllmAttention"):
        super().__init__(attn)
        self._multi_ctas_kv_counter: Optional[torch.Tensor] = None
        # Construct lazily: the layer can be initialized on meta before CUDA is ready.
        self._attention_ops = _AttentionOpCache()

    def attention_op(self, params: FmhaParams) -> "thop.AttentionOp":
        config = StaticAttentionConfig.from_params(
            params, skip_correction_threshold=self.attn.skip_correction_threshold
        )
        return self._attention_ops.get(config, _phase_query(params).device)

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

        Builds native FMHA parameters and dispatches to the phased
        ``AttentionOp.run_context`` / ``run_generation`` / ``run_mla_generation``
        ops. Direct callers supply the flat attention arguments and workspace.
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

        effective_beam_width = 1 if is_cross else beam_width
        sizing_num_contexts = num_contexts if num_ctx_tokens > 0 and not is_gen_only else 0
        params = interface.FmhaParams._from_arguments(
            arguments,
            qkv_or_q=q,
            layer_idx=local_layer_idx,
            num_seqs=sizing_num_contexts,
            num_requests=sizing_num_contexts,
            num_tokens=num_ctx_tokens if sizing_num_contexts > 0 else 0,
            max_num_sequences=max_num_sequences or max_num_requests,
            beam_width=effective_beam_width,
            multi_ctas_kv_counter=multi_ctas_kv_counter,
            fwd=_legacy_forward_args(
                arguments,
                chunked_prefill_buffer_batch_size=chunked_prefill_buffer_batch_size or 1,
            ),
            rotary_embedding_base=rope_base,
            rotary_embedding_scale_type=rope_scale_type,
            rotary_embedding_scale=rope_scale,
            rotary_embedding_short_mscale=rope_short_m_scale,
            rotary_embedding_long_mscale=rope_long_m_scale,
            rotary_embedding_max_positions=rope_max_positions,
            rotary_embedding_original_max_positions=rope_original_max_positions,
            num_sparse_topk=num_sparse_topk or 0,
            cyclic_attention_window_size=attention_window_size,
            max_attention_window_size=(
                attention_window_size
                if effective_beam_width == 1 or cache_indirection is None
                else cache_indirection.size(2)
            ),
        )
        tp = params.to_op_params()
        op = cls._compat_attention_ops.get(
            StaticAttentionConfig.from_legacy_arguments(arguments), arguments["q"].device
        )

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
            if max_context_q_len_override is not None:
                max_context_q_len = int(host_context_lengths[:num_contexts].max())
                max_past_kv_len = int(host_past_key_value_lengths[:num_contexts].max())
                override = int(max_context_q_len_override)
                if override < max_context_q_len or override < max_past_kv_len:
                    raise ValueError(
                        f"max_context_q_len_override ({override}) must be >= the computed max "
                        f"context q length ({max_context_q_len}) and max past kv length "
                        f"({max_past_kv_len})."
                    )
            tp.qkv_or_q = q[:num_ctx_tokens]
            if k is not None:
                tp.k = k[:num_ctx_tokens]
            if v is not None:
                tp.v = v[:num_ctx_tokens]
            tp.output = output[:num_ctx_tokens]
            tp.sequence_length = sequence_length[:num_contexts]
            tp.context_lengths = context_lengths[:num_contexts]
            tp.seq_offset = 0
            tp.num_seqs = num_contexts
            tp.num_requests = num_contexts
            tp.token_offset = 0
            tp.num_tokens = num_ctx_tokens
            op.run_context(tp)

        if num_generations > 0 and not is_ctx_only:
            # Native preparation mutates its carrier; each phase needs fresh parameters.
            if num_contexts > 0 and not is_gen_only:
                tp = params.to_op_params()
            seq_offset = num_contexts
            tp.qkv_or_q = q[num_ctx_tokens:]
            if k is not None:
                tp.k = k[num_ctx_tokens:]
            if v is not None:
                tp.v = v[num_ctx_tokens:]
            tp.output = output[num_ctx_tokens:]
            tp.sequence_length = sequence_length[seq_offset:]
            tp.context_lengths = context_lengths[seq_offset:]
            tp.seq_offset = seq_offset
            tp.num_seqs = num_generations
            tp.num_requests = num_generations // effective_beam_width
            # The tensors above are phase-local; token_offset only indexes the whole-batch
            # FP4 scaling-factor output.
            tp.token_offset = num_ctx_tokens
            tp.num_tokens = num_gen_tokens
            if is_mla_enable:
                op.run_mla_generation(tp)
            else:
                op.run_generation(tp)

    def _to_op_params(self, params: FmhaParams) -> "thop.FmhaParams":
        """Validate and lower one phase's Python parameters.

        The native struct is the only place these values are gathered: FmhaParams keeps
        the phase slice, and the layer and batch state stay behind `attn` and `meta`
        rather than being mirrored into a second carrier.
        """
        fwd, attn, meta = params.fwd, params.attn, params.meta
        if fwd is None:
            raise RuntimeError("FallbackFmha requires forward args.")
        if params.output is None:
            raise RuntimeError("FallbackFmha requires output.")
        if params.workspace is None:
            raise RuntimeError("FallbackFmha requires workspace.")
        query = _phase_query(params)

        tp = thop.FmhaParams()
        # Apply phase-local tensors and extents after the shared batch state.
        build_op_params(tp, meta, params)

        tp.qkv_or_q = query
        tp.sequence_length = params.sequence_lengths
        # Direct phase callers may omit the prompt-length view.
        tp.context_lengths = (
            params.context_lengths
            if params.context_lengths is not None
            else meta.prompt_lens_cuda_runtime[params.seq_offset :]
        )
        tp.num_seqs = params.batch_size
        # Encoder KV is request-scoped; each decoder beam reads it independently.
        tp.beam_width = meta.effective_beam_width
        tp.num_requests = params.batch_size if meta.is_cross else params.num_requests
        tp.host_past_key_value_lengths = meta.kv_lens_runtime
        tp.host_context_lengths = meta.prompt_lens_cpu_runtime
        tp.max_context_length = meta.max_context_length
        tp.max_num_requests = meta.max_num_requests
        tp.max_num_sequences = meta.max_num_sequences or meta.max_num_requests
        tp.layer_idx = attn.layer_idx
        tp.local_layer_idx = attn.get_local_layer_idx(meta)

        # Leave a native field alone when the source is None: it already holds the empty
        # value, and its setter takes the field's own type, as build_op_params assumes.
        for name, value in (
            ("k", params.key_input),
            ("v", params.value_input),
            ("host_kv_cache_pool_pointers", meta.host_kv_cache_pool_pointers),
            ("host_kv_cache_pool_mapping", meta.host_kv_cache_pool_mapping),
            ("cache_indirection", meta.cache_indirection),
            ("spec_decoding_target_max_draft_tokens", meta.max_total_draft_tokens),
            ("attention_chunk_size", attn.attention_chunk_size),
            ("rotary_inv_freq", attn.rotary_inv_freq),
            ("rotary_cos_sin", attn.rotary_cos_sin),
            ("multi_ctas_kv_counter", self._multi_ctas_kv_counter),
        ):
            if value is not None:
                setattr(tp, name, value)

        tp.has_fp8_kv_cache = bool(getattr(attn, "has_fp8_kv_cache", False))

        rope_params = attn.rope_params
        if rope_params is not None:
            # `rotary_embedding_dim` is layer configuration and reaches the op through
            # StaticAttentionConfig, so it is not resent here.
            tp.rotary_embedding_base = rope_params.theta
            tp.rotary_embedding_scale_type = rope_params.scale_type
            tp.rotary_embedding_scale = rope_params.scale
            tp.rotary_embedding_short_mscale = rope_params.short_m_scale
            tp.rotary_embedding_long_mscale = rope_params.long_m_scale
            tp.rotary_embedding_max_positions = rope_params.max_positions
            tp.rotary_embedding_original_max_positions = rope_params.original_max_positions
        return tp

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
        # The native sizing call wants a parameter carrier, but the phase split has not
        # run yet: build one covering the whole batch and point the phase counters at the
        # context extent, which is what bounds the context workspace.
        attn = self.attn
        output = forward_args.output
        if output is None:
            raise RuntimeError("FallbackFmha requires output.")
        num_tokens = q.size(0)
        is_gen_only = forward_args.attention_input_type == AttentionInputType.generation_only
        num_gen_tokens = num_tokens if is_gen_only else num_tokens - metadata.num_ctx_tokens
        out_head_size = self.generation_out_head_size if is_gen_only else self.context_out_head_size
        attention_window_size = forward_args.attention_window_size
        cache_indirection = metadata.cache_indirection
        max_attention_window_size = (
            attention_window_size
            if metadata.effective_beam_width == 1
            else (
                cache_indirection.size(2)
                if cache_indirection is not None
                else attention_window_size
            )
        )
        is_fused_qkv = forward_args.is_fused_qkv
        params = FmhaParams(
            attn=attn,
            meta=metadata,
            fwd=forward_args,
            workspace=workspace,
            qkv_input=q if is_fused_qkv else None,
            query_input=None if is_fused_qkv else q,
            key_input=k,
            value_input=v,
            # NVFP4 packs two logical values per uint8, so its buffer has no logical 3-D
            # shape; only the dtype matters here.
            output=(
                output
                if output.dtype == torch.uint8
                else output.view(num_tokens, attn.num_heads, out_head_size)
            ),
            sequence_lengths=metadata.kv_lens_cuda_runtime,
            context_lengths=metadata.prompt_lens_cuda_runtime,
            max_attention_window_size=max_attention_window_size,
            cyclic_attention_window_size=attention_window_size,
            tokens_per_block=(
                metadata.tokens_per_block if metadata.tokens_per_block is not None else 64
            ),
            kv_factor=self.kv_factor,
            is_cross=metadata.is_cross,
        )
        _set_context_workspace_shape(
            params,
            num_contexts=0 if is_gen_only else metadata.num_contexts,
            num_ctx_tokens=0 if is_gen_only else metadata.num_ctx_tokens,
        )
        tp = self._to_op_params(params)

        kv_cache_block_offsets = metadata.kv_cache_block_offsets
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
        self.attention_op(params).run_context(self._to_op_params(params))

    @torch.compiler.disable
    def run_mla_context(self, params: FmhaParams) -> None:
        self.attention_op(params).run_context(self._to_op_params(params))

    @torch.compiler.disable
    def run_generation(self, params: FmhaParams) -> None:
        self.attention_op(params).run_generation(self._to_op_params(params))

    @torch.compiler.disable
    def run_mla_generation(self, params: FmhaParams) -> None:
        self.attention_op(params).run_mla_generation(self._to_op_params(params))
