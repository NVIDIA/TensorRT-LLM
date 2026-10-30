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
from typing import TYPE_CHECKING, ClassVar, Mapping, Optional

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
    RopeParams,
)
from tensorrt_llm._utils import get_sm_version, is_sm_120f
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

from ..sparse.params import SparseRuntimeParams
from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


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


class FallbackFmha(PhasedFmha):
    """Fallback FMHA implementation over the phased TRT-LLM thop ops."""

    REQUIRES_PAGED_KV = False
    # Native attention consumes pool pointers directly; sparse managers may have
    # no Role.KEY buffer from which to query a dense block extent.
    NEEDS_BLOCK_EXTENT = False
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
    # exact-parameter check is the context preparation check in
    # thop/attentionOp.cpp; this gate exists to move the
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

    @classmethod
    def _is_available(cls, attn: "TrtllmAttention") -> bool:
        sparse_algorithm = getattr(attn.sparse_params, "algorithm", None)
        if sparse_algorithm in ("deepseek_v4", "dsa"):
            if getattr(attn, "kv_cache_dtype", None) == "fp8_ds_mla":
                return False
            if is_sm_120f():
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
        del k, v
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
            # Extra page sizes are supported by TRTLLM-Gen, not the thop
            # paged-context kernel. Refuse before the kernel can read KV.
            extra_page_sizes = getattr(
                metadata.kv_cache_manager, "trtllm_gen_extra_tokens_per_block", ()
            )
            if (
                extra_page_sizes
                and phase != FmhaPhase.GENERATION
                and metadata.use_paged_context_fmha
                and metadata.tokens_per_block in extra_page_sizes
            ):
                raise RuntimeError(
                    f"FallbackFmha cannot serve paged prefill with "
                    f"{metadata.tokens_per_block}-token KV pages. "
                    "Use a supported TRTLLM-Gen configuration or disable "
                    "chunked prefill, KV reuse and speculative decoding."
                )
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
        if not (
            self.attn.is_mla_enable
            or forward_args.is_fused_qkv
            or metadata.is_cross
            or forward_args.sage_attn_num_elts_per_blk_q > 0
            or forward_args.sage_attn_num_elts_per_blk_k > 0
            or forward_args.sage_attn_num_elts_per_blk_v > 0
        ):
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
            thop.run_context(params)

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
            thop.run_generation(params)

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        """The native op sizes and grows the workspace itself, inside each phase call."""
        if forward_args.sage_attn_smooth_v:
            logger.warning_once(
                "The attention op quantizes V without the V smoothing this SageAttention "
                "recipe requests.",
                key="fallback_fmha_sage_attn_smooth_v",
            )

    # Keep nanobind calls eager, including when CombinedFmha delegates individual phases.
    @torch.compiler.disable
    def run_context(self, params: FmhaParams) -> None:
        thop.run_context(params)

    @torch.compiler.disable
    def run_mla_context(self, params: FmhaParams) -> None:
        thop.run_context(params)

    @torch.compiler.disable
    def run_generation(self, params: FmhaParams) -> None:
        thop.run_generation(params)

    @torch.compiler.disable
    def run_mla_generation(self, params: FmhaParams) -> None:
        # The native op dispatches to the MLA generation path from the layer configuration.
        thop.run_generation(params)
