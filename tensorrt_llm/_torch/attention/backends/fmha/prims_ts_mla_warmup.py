# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup-only planning for dense MLA profiles already selected by PrimTS."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.attention.backends.prims_ts.context import _ContextPlanGeometry
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import _MLADecodePlanState
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


def _bucket_minima(limit: int) -> list[int]:
    """Smallest positive length in each power-of-two bucket up to limit."""
    if limit < 1:
        return []
    return [1] + [(1 << exponent) + 1 for exponent in range((limit - 1).bit_length())]


def _context_profiles(
    *,
    max_num_tokens: int,
    max_seq_len: int,
    cached_context: bool,
    chunked_prefill: bool,
    chunk_size: int,
    chunk_batch_size: int,
    capacities: Callable[[int, int], tuple[int, int]],
) -> list[tuple[int, int, str, bool]]:
    """Enumerate the normalized fresh, cached-prefix and current-chunk envelope."""
    queries = _bucket_minima(min(max_num_tokens, max_seq_len))
    profiles = {(*capacities(query, query), "causal", False) for query in queries}
    if not cached_context:
        return sorted(profiles)
    if chunked_prefill:
        if chunk_size <= 0 or chunk_batch_size <= 0:
            raise ValueError("MLA chunked-prefill warmup requires a positive staging capacity.")
        max_cached_kv = min(max_seq_len - 1, chunk_size * chunk_batch_size)
        for query in queries:
            if query < max_seq_len:
                profiles.add((*capacities(query, query), "causal", True))
            # Runtime takes these maxima independently across requests, so
            # their sum need not fit within one request's sequence limit.
            for cached_kv in _bucket_minima(max_cached_kv):
                profiles.add((*capacities(query, cached_kv), "dense", True))
    else:
        for query in queries:
            for kv_length in _bucket_minima(max_seq_len):
                if query <= kv_length:
                    profiles.add((*capacities(query, kv_length), "causal", False))
    return sorted(profiles)


class PrimsTSDenseMlaWarmup:
    """Model-lifetime compile deduplication, with no retained runtime workspace.

    Context covers the configured normalized envelope. Decode expands batch
    counts only for widths and modes observed in this model's startup plans;
    arbitrary future proposal widths from variable drafters are not inferred.
    """

    def __init__(self) -> None:
        self._completed: set[Hashable] = set()

    def run(
        self,
        model: torch.nn.Module,
        metadata: TrtllmAttentionMetadata | None,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
    ) -> tuple[int, int]:
        """Load selected providers' missing profiles outside CUDA graph capture."""
        import torch

        from tensorrt_llm._torch.attention.backends.trtllm import (
            TrtllmAttention,
            TrtllmAttentionMetadata,
        )
        from tensorrt_llm._torch.attention.mla import MLA
        from tensorrt_llm.logger import logger

        from .prims_ts import PrimsTSFmha

        if not isinstance(metadata, TrtllmAttentionMetadata):
            return 0, 0
        context_states = []
        decode_states: list[_MLADecodePlanState] = []
        for module in model.modules():
            if not isinstance(module, MLA) or module.sparse_attn_hooks is not None:
                continue
            for backend in (module.mha, module.mqa):
                if not isinstance(backend, TrtllmAttention):
                    continue
                for library in backend._fmha_manager.fmha_libs:
                    if not isinstance(library, PrimsTSFmha):
                        continue
                    for context_wrapper in library._mla_context_wrappers.values():
                        context_state = context_wrapper._plan_state
                        if (
                            context_state is not None
                            and context_state.geometry.device.type == "cuda"
                        ):
                            context_states.append(context_state)
                    for decode_wrapper in library._mla_decode_wrappers.values():
                        decode_state = decode_wrapper._plan_state
                        if decode_state is not None and decode_state.device.type == "cuda":
                            decode_states.append(decode_state)
        if not context_states and not decode_states:
            return 0, 0
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS dense MLA startup planning cannot run during capture.")

        contexts = decodes = 0
        widths = set()
        for state in context_states:
            contexts += self._prewarm_context(
                state.geometry,
                metadata,
                max_num_tokens=max_num_tokens,
                max_seq_len=max_seq_len,
                softmax_scale=state.softmax_scale,
                output_scale=state.output_scale,
            )
        for state in decode_states:
            widths.add(state.max_q_len)
            decodes += self._prewarm_decode(
                state, max_batch_size=max_batch_size, max_num_tokens=max_num_tokens
            )
        logger.info(
            f"PrimTS dense MLA prewarm loaded {contexts} context profiles and {decodes} "
            f"decode topologies; observed decode query widths: {sorted(widths)}."
        )
        return contexts, decodes

    def _prewarm_context(
        self,
        geometry: _ContextPlanGeometry,
        metadata: TrtllmAttentionMetadata,
        *,
        max_num_tokens: int,
        max_seq_len: int,
        softmax_scale: float | torch.Tensor = 1.0,
        output_scale: float | torch.Tensor = 1.0,
    ) -> int:
        from tensorrt_llm._torch.attention.backends.prims_ts import context
        from tensorrt_llm._torch.attention.backends.prims_ts._scales import scale_types

        from .prims_ts import _mla_context_capacities

        capacities = partial(
            _mla_context_capacities,
            head_dim=geometry.head_dim,
            head_dim_vo=geometry.head_dim_vo,
            qk_dtype=geometry.qk_dtype,
            pv_dtype=geometry.pv_dtype,
            output_dtype=geometry.out_dtype,
            packed=geometry.packed_query,
            uniform_packed_lengths=geometry.uniform_packed_lengths,
            mask_type=geometry.mask_type,
            window_left=geometry.window_left,
        )
        # Only the proven query-paired layout has a bounded normalized mesh.
        if geometry.num_qo_heads != geometry.num_kv_heads or capacities(1, 1) != (256, 128):
            return 0
        features = metadata.runtime_features
        profiles = _context_profiles(
            max_num_tokens=max_num_tokens,
            max_seq_len=max_seq_len,
            cached_context=metadata.enable_context_mla_with_cached_kv,
            chunked_prefill=features.chunked_prefill,
            chunk_size=features.chunk_size,
            chunk_batch_size=features.chunked_prefill_buffer_batch_size,
            capacities=capacities,
        )
        loaded = 0
        for query_capacity, kv_capacity, mask, stats in profiles:
            kwargs = dict(
                device=geometry.device,
                batch_size=1,
                max_q_len=query_capacity,
                max_kv_len=kv_capacity,
                num_qo_heads=geometry.num_qo_heads,
                num_kv_heads=geometry.num_kv_heads,
                head_dim=geometry.head_dim,
                head_dim_vo=geometry.head_dim_vo,
                packed_query=True,
                mask_type=mask,
                window_left=-1,
                store_softmax_stats=stats,
            )
            candidate = context._resolve_context_plan_geometry(
                **kwargs,
                qk_dtype=geometry.qk_dtype,
                pv_dtype=geometry.pv_dtype,
                out_dtype=geometry.out_dtype,
            )
            signature = context._context_compile_spec(
                candidate, scale_types=scale_types(softmax_scale, output_scale)
            )
            if signature in self._completed:
                continue
            # Normal plan() loads real callables. Probe/precompile-only modes
            # must not populate the context compiler's functools cache.
            context.BatchPrefillTSWrapper().plan(
                **kwargs,
                q_dtype=geometry.qk_dtype,
                k_dtype=geometry.qk_dtype,
                v_dtype=geometry.pv_dtype,
                out_dtype=geometry.out_dtype,
                softmax_scale=softmax_scale,
                output_scale=output_scale,
            )
            self._completed.add(signature)
            loaded += 1
        return loaded

    def _prewarm_decode(
        self,
        state: _MLADecodePlanState,
        *,
        max_batch_size: int,
        max_num_tokens: int,
    ) -> int:
        from tensorrt_llm._torch.attention.backends.prims_ts import mla_decode
        from tensorrt_llm._torch.attention.backends.prims_ts._scales import scale_types

        query_width = state.max_q_len
        max_batch = min(max_batch_size, max_num_tokens // query_width)
        representatives = {}
        q_dtype_key = mla_decode._dtype_key(state.q_dtype)
        kv_dtype_key = mla_decode._dtype_key(state.kv_dtype)
        output_dtype_key = mla_decode._dtype_key(state.out_dtype)
        for batch_size in range(1, max_batch + 1):
            launch = mla_decode._resolve_mla_decode_launch_spec(
                state.device.index,
                batch_size,
                state.num_qo_heads,
                state.kv_lora_rank,
                state.qk_rope_head_dim,
                state.page_size,
                state.max_kv_len,
                q_dtype_key,
                kv_dtype_key,
                output_dtype_key,
                state.mask_type,
                query_width,
                softmax_rescale_threshold_log2=state.softmax_rescale_threshold_log2,
                store_softmax_stats=state.store_softmax_stats,
                use_kv_bounds=state.use_kv_bounds,
            )
            signature = mla_decode._make_mla_decode_compile_spec(
                launch,
                device_index=state.device.index,
                num_qo_heads=state.num_qo_heads,
                kv_lora_rank=state.kv_lora_rank,
                qk_rope_head_dim=state.qk_rope_head_dim,
                page_size=state.page_size,
                q_dtype_key=q_dtype_key,
                out_dtype_key=output_dtype_key,
                max_q_len=query_width,
                packed_query=state.packed_query,
                runtime_kv_lengths=True,
                write_final_lse=state.lse_mode != "none",
                lse_mode=state.lse_mode,
                scale_types=(*scale_types(state.softmax_scale, state.output_scale), False),
                store_softmax_stats=state.store_softmax_stats,
                use_kv_bounds=state.use_kv_bounds,
            )
            if signature not in self._completed:
                representatives[signature] = batch_size
        for signature, batch_size in representatives.items():
            mla_decode.BatchMLADecodePagedTSWrapper().plan(
                state.device,
                batch_size,
                state.num_qo_heads,
                state.kv_lora_rank,
                state.qk_rope_head_dim,
                state.page_size,
                state.max_kv_len,
                max_q_len=query_width,
                packed_query=state.packed_query,
                q_dtype=state.q_dtype,
                kv_dtype=state.kv_dtype,
                out_dtype=state.out_dtype,
                mask_type=state.mask_type,
                softmax_scale=state.softmax_scale,
                output_scale=state.output_scale,
                lse_mode=state.lse_mode,
                tuning=mla_decode.MLAAttentionTuning(
                    softmax_rescale_threshold_log2=state.softmax_rescale_threshold_log2
                ),
                store_softmax_stats=state.store_softmax_stats,
                use_kv_bounds=state.use_kv_bounds,
            )
            self._completed.add(signature)
        return len(representatives)
