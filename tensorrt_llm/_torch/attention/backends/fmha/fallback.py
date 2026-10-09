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
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

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
    # runners hang off the class instead, reused across calls and captures.
    _compat_attention_ops: ClassVar[_AttentionOpCache] = _AttentionOpCache()
    supports_skip_correction = True
    supports_workspace_reclamation = True

    # Head sizes whose fused context FMHA kernel is proven absent for at
    # least one KV cache dtype, per SM version. This is a blocklist of
    # proven-problematic cells, not a support matrix: sourced from the
    # runtime fallback warning "Fall back to unfused MHA ... headSize = 64
    # ... in sm_103" and the silent-wrong-answer incident it caused. Inside
    # a blocklisted cell, the ``fused_context_fmha_kernel_exists`` native
    # lookup decides per KV dtype and page size which configurations the
    # running build can serve. Outside it, no probe runs: the lookup is a
    # fixed-convention diagnostic (dense causal Q_PAGED_KV, Q and output
    # precision inferred from the KV precision) that does not model every
    # configuration the op can run (MLA, cross attention, 16-bit context
    # math over an FP8 KV cache), so probing unconditionally would refuse
    # configurations the op serves correctly. The authoritative
    # exact-parameter check is the op-level refusal in
    # thop/attentionOp.cpp's AttentionOp constructor; this gate exists to move the
    # proven-problematic cells to FMHA dispatch with a remedial error.
    # Extend as further combinations are proven.
    CONTEXT_FMHA_ABSENT_HEAD_DIMS: ClassVar[dict[int, tuple[int, ...]]] = {
        103: (64,),
    }

    @classmethod
    def _validate_paged_context_fmha(cls, metadata: "TrtllmAttentionMetadata") -> None:
        """Refuse paged-context FMHA when no kernel exists for this config.

        When ``use_paged_context_fmha`` is enabled but the fused context FMHA
        kernel is absent for this SM/head-size combination, attentionOp.cpp
        falls back to unfused MHA whose context path builds K/V from the
        current chunk only: the cached prefix is DROPPED from attention and
        then OVERWRITTEN by the chunk's write-back. Any request with a
        non-zero cached length -- block reuse, partial reuse, chunked
        prefill, speculative draft tokens -- returns a plausible wrong answer
        with no error. That fallback happens inside the C++ op, after
        selection has already committed to this library, so ``_is_supported``
        calls this check for every batch with a context phase and raises
        instead of returning False: this library is last in the registry, so
        the raise skips no other library, and it replaces the silent wrong
        answer with an error naming the cause and the remedy.

        Inside a blocklisted (SM, head_dim) cell the decision is per KV
        dtype and page size, made by asking the build what it contains via
        the ``fused_context_fmha_kernel_exists`` native lookup: a present
        kernel admits the configuration, an absent one refuses it. Nothing
        about kernel presence is hand-maintained. The check fails closed
        wherever it cannot be made: managers without a ``dtype`` or
        ``tokens_per_block`` attribute, and builds whose bindings predate
        the lookup (which also predate the op-level refusal, so passing
        unverified would reintroduce the silent corruption there).
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
        if not absent:
            return
        kv_dtype = getattr(manager, "dtype", None)
        tokens_per_block = getattr(manager, "tokens_per_block", None)
        if kv_dtype is not None and tokens_per_block is not None:
            kernel_exists = getattr(thop, "fused_context_fmha_kernel_exists", None)
            if kernel_exists is None:
                raise RuntimeError(
                    f"Paged-context FMHA with head_dim {absent} on SM {sm} "
                    f"requires confirming the fused context kernel against "
                    f"this build via the fused_context_fmha_kernel_exists "
                    f"lookup, but this build's bindings predate it. Bindings "
                    f"that old also predate the op-level paged-context "
                    f"refusal, so an unverified pass would risk a silent "
                    f"fall back to unfused MHA that corrupts the cached "
                    f"prefix; the configuration stays refused. Rebuild the "
                    f"bindings, or use the FlashInfer attention backend."
                )
            # Probe with every output precision the op can pair with this
            # KV precision. For an FP8 KV cache the op quantizes Q to FP8
            # but keeps dataTypeOut at the activation dtype (BF16/FP16)
            # unless FP8 attention output is enabled, and neither the
            # activation dtype nor that per-module flag is visible to this
            # metadata-only check, so one present variant admits the
            # cell; if the variant the model actually needs is the absent
            # one, the exact-parameter refusal in the AttentionOp constructor still
            # raises. An NVFP4 KV cache is read by the FP8-output kernels,
            # and a 16-bit KV cache runs matched output, mirroring the
            # binding's probe convention in
            # test_context_fmha_kernel_presence.py. Dtypes the lookup does
            # not model report absent, so they stay refused.
            if kv_dtype == DataType.FP8:
                probe_output_dtypes = (DataType.BF16, DataType.HALF, DataType.FP8)
            elif kv_dtype == DataType.NVFP4:
                probe_output_dtypes = (DataType.FP8,)
            else:
                probe_output_dtypes = (kv_dtype,)
            if all(
                any(
                    kernel_exists(
                        head_size=dim,
                        kv_cache_dtype=kv_dtype,
                        tokens_per_block=tokens_per_block,
                        output_dtype=probe_output_dtype,
                    )
                    for probe_output_dtype in probe_output_dtypes
                )
                for dim in absent
            ):
                logger.info_once(
                    f"Paged-context FMHA enabled for head_dim {absent} on SM "
                    f"{sm}: this build's fused context FMHA kernel is present "
                    f"for KV cache dtype {kv_dtype} at {tokens_per_block} "
                    f"tokens per block.",
                    key=f"paged_context_fmha_present_{sm}_{absent}_{kv_dtype}_{tokens_per_block}",
                )
                return
            cause = (
                f"this build contains no fused context FMHA kernel for that "
                f"combination with KV cache dtype {kv_dtype} at "
                f"{tokens_per_block} tokens per block"
            )
        else:
            missing = [
                name
                for name, value in (("dtype", kv_dtype), ("tokens_per_block", tokens_per_block))
                if value is None
            ]
            cause = (
                f"the KV cache manager exposes no {' or '.join(missing)}, so "
                f"the fused context FMHA kernel cannot be confirmed present "
                f"(the check fails closed)"
            )
        features = [
            name
            for name in ("chunked_prefill", "cache_reuse", "has_speculative_draft_tokens")
            if getattr(metadata.runtime_features, name, False)
        ]
        raise RuntimeError(
            f"{'/'.join(features)} requires attending to cached KV during "
            f"the context phase (use_paged_context_fmha), but for head_dim "
            f"{absent} on SM {sm} {cause}. The unfused fallback silently "
            f"drops and corrupts the cached prefix, producing plausible "
            f"wrong answers, so the configuration is refused. A build whose "
            f"--cuda_architectures does not name SM {sm} carries no kernels "
            f"for SM {sm} at all and lands here too; check the architecture "
            f"list first. Otherwise use the FlashInfer attention backend, or "
            f"disable KV block reuse, chunked prefill and speculative "
            f"decoding for this model."
        )

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
        # A context phase served by the thop op with use_paged_context_fmha
        # enabled would silently corrupt the cached prefix wherever the fused
        # context kernel is absent. Raise rather than return False: no
        # library follows this one, and the error names the cause and the
        # remedy instead of the generic no-library message. Generation-only
        # batches never run the context path, so they pass; whether a batch
        # has a context phase is part of the FMHA cache key
        # (``context_batch_size``), so the admitted result cannot be reused
        # for a context batch.
        if metadata.num_contexts > 0:
            self._validate_paged_context_fmha(metadata)
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
            tp.output = output if enable_dsv4_epilogue_fusion else output[:num_ctx_tokens]
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
            tp.output = output if enable_dsv4_epilogue_fusion else output[num_ctx_tokens:]
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
            # Workspace sizing only needs the output dtype; preserve the caller's layout.
            output=output,
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
