# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup-only planning for dense MLA profiles already selected by PrimTS."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


class _ScaleKind(Enum):
    DEVICE = auto()


def _scale_profile(scale: float | torch.Tensor) -> float | _ScaleKind:
    """Record the public scale operand kind without retaining its storage."""
    return _ScaleKind.DEVICE if isinstance(scale, torch.Tensor) else scale


@dataclass(frozen=True, kw_only=True)
class _MlaContextPlan:
    device: torch.device
    batch_size: int
    max_q_len: int
    max_kv_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    head_dim_vo: int
    q_dtype: torch.dtype
    k_dtype: torch.dtype
    v_dtype: torch.dtype
    out_dtype: torch.dtype
    packed_query: bool
    mask_type: Literal["causal", "dense"]
    softmax_scale: float | _ScaleKind
    output_scale: float | _ScaleKind
    store_softmax_stats: bool
    window_left: int = -1


@dataclass(frozen=True, kw_only=True)
class _MlaDecodePlan:
    device: torch.device
    batch_size: int
    num_qo_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    max_kv_len: int
    max_q_len: int
    packed_query: bool
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    out_dtype: torch.dtype
    mask_type: Literal["causal", "dense"]
    store_softmax_stats: bool
    use_kv_bounds: bool


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
) -> list[tuple[int, int, Literal["causal", "dense"], bool]]:
    """Enumerate the normalized fresh, cached-prefix and current-chunk envelope."""
    queries = _bucket_minima(min(max_num_tokens, max_seq_len))
    profiles: set[tuple[int, int, Literal["causal", "dense"], bool]] = {
        (*capacities(query, query), "causal", False) for query in queries
    }
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


def _context_plans(
    observed: _MlaContextPlan,
    metadata: TrtllmAttentionMetadata,
    *,
    max_num_tokens: int,
    max_seq_len: int,
) -> Iterator[_MlaContextPlan]:
    from .prims_ts import _mla_context_capacities

    capacities = partial(
        _mla_context_capacities,
        head_dim=observed.head_dim,
        head_dim_vo=observed.head_dim_vo,
        qk_dtype=observed.q_dtype,
        pv_dtype=observed.v_dtype,
        output_dtype=observed.out_dtype,
        packed=observed.packed_query,
        uniform_packed_lengths=False,
        mask_type=observed.mask_type,
        window_left=observed.window_left,
    )
    # Only the proven query-paired layout has a bounded normalized mesh.
    if observed.num_qo_heads != observed.num_kv_heads or capacities(1, 1) != (256, 128):
        return
    features = metadata.runtime_features
    for query_capacity, kv_capacity, mask, stats in _context_profiles(
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        cached_context=metadata.enable_context_mla_with_cached_kv,
        chunked_prefill=features.chunked_prefill,
        chunk_size=features.chunk_size,
        chunk_batch_size=features.chunked_prefill_buffer_batch_size,
        capacities=capacities,
    ):
        yield replace(
            observed,
            batch_size=1,
            max_q_len=query_capacity,
            max_kv_len=kv_capacity,
            mask_type=mask,
            store_softmax_stats=stats,
        )


def _plan_context(profile: _MlaContextPlan) -> None:
    from tensorrt_llm._torch.attention.backends.prims_ts.context import BatchPrefillTSWrapper

    kwargs = asdict(profile)
    wrapper = None
    try:
        for name in ("softmax_scale", "output_scale"):
            if kwargs[name] is _ScaleKind.DEVICE:
                kwargs[name] = torch.ones((), dtype=torch.float32, device=profile.device)
        wrapper = BatchPrefillTSWrapper()
        wrapper.plan(**kwargs)
    finally:
        # Neither the temporary scale operands nor the context metadata are
        # part of the model-lifetime completion key.
        wrapper = None
        kwargs.clear()


def _plan_decode(profile: _MlaDecodePlan, workspace: torch.Tensor) -> None:
    from tensorrt_llm._torch.attention.backends.prims_ts.mla_decode import (
        BatchMLADecodePagedTSWrapper,
    )

    wrapper = BatchMLADecodePagedTSWrapper()
    try:
        wrapper.plan(**asdict(profile), workspace_buffer=workspace)
    finally:
        del wrapper, workspace


class PrimsTSDenseMlaWarmup:
    """Deduplicate public startup profiles without retaining runtime workspace.

    Context covers the configured normalized envelope. Decode expands batch
    counts only for widths and modes observed in this model's startup plans;
    arbitrary future proposal widths from variable drafters are not inferred.
    PrimTS owns compiled-kernel deduplication behind its public plan methods.
    """

    def __init__(self) -> None:
        self._completed: set[_MlaContextPlan | _MlaDecodePlan] = set()

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
        from tensorrt_llm._torch.attention.backends.prims_ts import (
            get_prims_ts_batch_mla_decode_workspace_size,
        )
        from tensorrt_llm._torch.attention.backends.trtllm import (
            TrtllmAttention,
            TrtllmAttentionMetadata,
        )
        from tensorrt_llm._torch.attention.mla import MLA
        from tensorrt_llm.logger import logger

        from .prims_ts import PrimsTSFmha

        if not isinstance(metadata, TrtllmAttentionMetadata):
            return 0, 0
        context_profiles: set[_MlaContextPlan] = set()
        decode_profiles: set[_MlaDecodePlan] = set()
        for module in model.modules():
            if not isinstance(module, MLA) or module.sparse_attn_hooks is not None:
                continue
            for backend in (module.mha, module.mqa):
                if not isinstance(backend, TrtllmAttention):
                    continue
                for library in backend._fmha_manager.fmha_libs:
                    if isinstance(library, PrimsTSFmha):
                        context_profiles.update(library._mla_context_plan_profiles)
                        decode_profiles.update(library._mla_decode_plan_profiles)
        context_profiles = {p for p in context_profiles if p.device.type == "cuda"}
        decode_profiles = {p for p in decode_profiles if p.device.type == "cuda"}
        if not context_profiles and not decode_profiles:
            return 0, 0
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS dense MLA startup planning cannot run during capture.")

        contexts = {
            candidate
            for observed in context_profiles
            for candidate in _context_plans(
                observed, metadata, max_num_tokens=max_num_tokens, max_seq_len=max_seq_len
            )
            if candidate not in self._completed
        }
        decodes = {
            replace(observed, batch_size=batch_size)
            for observed in decode_profiles
            for batch_size in range(
                1, min(max_batch_size, max_num_tokens // observed.max_q_len) + 1
            )
        } - self._completed
        workspace_sizes: dict[torch.device, int] = {}
        for profile in decodes:
            size = get_prims_ts_batch_mla_decode_workspace_size(
                batch_size=profile.batch_size,
                num_qo_heads=profile.num_qo_heads,
                kv_lora_rank=profile.kv_lora_rank,
                qk_rope_head_dim=profile.qk_rope_head_dim,
                page_size=profile.page_size,
                max_kv_len=profile.max_kv_len,
                max_q_len=profile.max_q_len,
                q_dtype=profile.q_dtype,
                kv_dtype=profile.kv_dtype,
                out_dtype=profile.out_dtype,
                mask_type=profile.mask_type,
                store_softmax_stats=profile.store_softmax_stats,
                use_kv_bounds=profile.use_kv_bounds,
                device=profile.device,
            )
            workspace_sizes[profile.device] = max(workspace_sizes.get(profile.device, 0), size)

        workspaces: dict[torch.device, torch.Tensor] = {}
        try:
            for device, size in workspace_sizes.items():
                workspaces[device] = torch.empty(size, dtype=torch.uint8, device=device)
            for context_profile in contexts:
                _plan_context(context_profile)
                self._completed.add(context_profile)
            for decode_profile in decodes:
                _plan_decode(decode_profile, workspaces[decode_profile.device])
                self._completed.add(decode_profile)
        finally:
            workspaces.clear()
        logger.info(
            f"PrimTS dense MLA prewarm loaded {len(contexts)} context profiles and {len(decodes)} "
            f"decode profiles; observed decode query widths: "
            f"{sorted({p.max_q_len for p in decode_profiles})}."
        )
        return len(contexts), len(decodes)
