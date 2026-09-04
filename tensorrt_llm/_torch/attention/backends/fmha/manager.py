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

"""FMHA library discovery, selection, and selection caching."""

from __future__ import annotations

import bisect
import functools
import inspect
import os
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
)
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger

from .combined import CombinedFmha
from .interface import Fmha, FmhaPhase
from .phased import PhasedFmha
from .registry import get_enabled_fmha_lib_classes

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


# Keep these general-purpose cache buckets aligned with the shape coverage used
# by TRTLLM-Gen JIT warmup in fmhaKernels.h. Adding the predecessor of every
# ceiling candidate prevents support changes at a candidate from aliasing with
# the interval immediately below it. Values above the largest grid point share
# the largest bucket.
_FMHA_CACHE_BATCH_SIZE_CANDIDATES: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    26,
    28,
    30,
    32,
    36,
    40,
    48,
    56,
    64,
    80,
    96,
    128,
    256,
    384,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
)
_FMHA_CACHE_SEQ_LEN_Q_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8, 32, 64, 128)


def _make_fmha_cache_grid(candidates: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                boundary
                for candidate in candidates
                for boundary in (candidate - 1, candidate)
                if boundary > 0
            }
        )
    )


_FMHA_CACHE_BATCH_SIZE_GRID: tuple[int, ...] = _make_fmha_cache_grid(
    _FMHA_CACHE_BATCH_SIZE_CANDIDATES
)
_FMHA_CACHE_SEQ_LEN_Q_GRID: tuple[int, ...] = _make_fmha_cache_grid(
    _FMHA_CACHE_SEQ_LEN_Q_CANDIDATES
)
_FMHA_CACHE_SANITY_CHECK_ENV = "TRTLLM_FMHA_CACHE_SANITY_CHECK"


class _FmhaCacheKey(NamedTuple):
    context_batch_size: int
    generation_batch_size: int
    generation_seq_len_q: int
    attention_mask_type: AttentionMaskType
    use_spec_decoding: bool
    is_cuda_graph: bool
    is_fused_qkv: bool
    update_kv_cache: bool
    # LoRA can change the effective output from packed NVFP4 to unpacked BF16
    # without changing the request shape. Keep those selection regimes apart.
    output_dtype: torch.dtype | None
    output_sf_dtype: torch.dtype | None


_FmhaCacheInputs = dict[str, str]
_FMHA_CACHE_INPUT_EXCLUDED_FIELDS = frozenset({"kv_cache_manager"})


def _should_snapshot_fmha_cache_field(value: object, name: str) -> bool:
    descriptor = inspect.getattr_static(type(value), name, None)
    return (
        not name.startswith("_")
        and name not in _FMHA_CACHE_INPUT_EXCLUDED_FIELDS
        and not isinstance(descriptor, (property, functools.cached_property))
    )


def _normalize_fmha_cache_grid_value(value: int, grid: tuple[int, ...]) -> int:
    if value <= 0:
        return 0
    index = bisect.bisect_left(grid, value)
    return grid[min(index, len(grid) - 1)]


def _is_fmha_cache_enabled() -> bool:
    # Selection policy may differ while tuning (currently for CuTe DSL MLA).
    # Keep all temporary selections out of the serving cache so future FMHAs
    # inherit the same cache boundary.
    from tensorrt_llm._torch.autotuner import AutoTuner

    autotuner = AutoTuner._instance
    return autotuner is None or not autotuner.is_tuning_mode


def _is_fmha_cache_sanity_check_enabled() -> bool:
    return os.environ.get(_FMHA_CACHE_SANITY_CHECK_ENV, "0") == "1"


def _snapshot_fmha_cache_value(
    snapshot: _FmhaCacheInputs,
    path: str,
    value: object,
    seen: set[int],
) -> None:
    """Record tensor shapes and primitive values without retaining inputs."""
    if isinstance(value, torch.Tensor):
        snapshot[path] = f"shape={tuple(value.shape)}"
    elif isinstance(value, Enum):
        snapshot[path] = f"{type(value).__name__}.{value.name}"
    elif isinstance(value, (torch.dtype, torch.device)):
        snapshot[path] = str(value)
    elif value is None or isinstance(value, (bool, int, float, str)):
        snapshot[path] = repr(value)
    elif is_dataclass(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        for value_field in fields(value):
            if _should_snapshot_fmha_cache_field(value, value_field.name):
                _snapshot_fmha_cache_value(
                    snapshot,
                    f"{path}.{value_field.name}",
                    getattr(value, value_field.name),
                    seen,
                )
    elif isinstance(value, (list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
        for index, item in enumerate(value):
            _snapshot_fmha_cache_value(snapshot, f"{path}[{index}]", item, seen)
    elif isinstance(value, dict):
        if id(value) in seen:
            return
        seen.add(id(value))
        for key in sorted(value, key=repr):
            _snapshot_fmha_cache_value(snapshot, f"{path}[{key!r}]", value[key], seen)


def _snapshot_fmha_cache_fields(
    snapshot: _FmhaCacheInputs,
    path: str,
    value: object,
    seen: set[int],
) -> None:
    seen.add(id(value))
    if is_dataclass(value):
        value_fields = (
            (value_field.name, getattr(value, value_field.name))
            for value_field in fields(value)
            if _should_snapshot_fmha_cache_field(value, value_field.name)
        )
    else:
        value_fields = (
            (name, field_value)
            for name, field_value in vars(value).items()
            if _should_snapshot_fmha_cache_field(value, name)
        )
    for name, field_value in value_fields:
        _snapshot_fmha_cache_value(snapshot, f"{path}.{name}", field_value, seen)


def _snapshot_fmha_cache_inputs(
    q: torch.Tensor,
    k: torch.Tensor | None,
    v: torch.Tensor | None,
    metadata: TrtllmAttentionMetadata,
    forward_args: AttentionForwardArgs,
) -> _FmhaCacheInputs:
    snapshot: _FmhaCacheInputs = {}
    seen: set[int] = set()
    _snapshot_fmha_cache_value(snapshot, "q", q, seen)
    _snapshot_fmha_cache_value(snapshot, "k", k, seen)
    _snapshot_fmha_cache_value(snapshot, "v", v, seen)
    _snapshot_fmha_cache_fields(snapshot, "metadata", metadata, seen)
    _snapshot_fmha_cache_fields(snapshot, "forward_args", forward_args, seen)
    return snapshot


def _format_fmha_cache_input_diff(
    cached: _FmhaCacheInputs | None,
    uncached: _FmhaCacheInputs,
) -> str:
    if cached is None:
        return (
            "  cached input snapshot unavailable; enable "
            f"{_FMHA_CACHE_SANITY_CHECK_ENV} before the cache entry is created"
        )

    missing = "<missing>"
    differences = []
    for path in sorted(cached.keys() | uncached.keys()):
        cached_value = cached.get(path, missing)
        uncached_value = uncached.get(path, missing)
        if cached_value != uncached_value:
            differences.append(f"  {path}: cached={cached_value}, uncached={uncached_value}")
    if not differences:
        return "  (no captured input differences)"
    return "\n".join(differences)


def _fmha_cache_values_match(cached: Fmha | None, uncached: Fmha | None) -> bool:
    if cached is uncached:
        return True
    if not (isinstance(cached, CombinedFmha) and isinstance(uncached, CombinedFmha)):
        return False
    return _fmha_cache_values_match(
        cached._get_context_impl(), uncached._get_context_impl()
    ) and _fmha_cache_values_match(cached._get_generation_impl(), uncached._get_generation_impl())


def _describe_fmha_cache_value(fmha: Fmha | None) -> str:
    if fmha is None:
        return "None"
    if isinstance(fmha, CombinedFmha):
        context = _describe_fmha_cache_value(fmha._get_context_impl())
        generation = _describe_fmha_cache_value(fmha._get_generation_impl())
        return f"CombinedFmha(context={context}, generation={generation})"
    return type(fmha).__name__


class FmhaManager:
    """Own the FMHA libraries, dispatch policy, and selection cache for one attention layer."""

    def __init__(self, attn: TrtllmAttention) -> None:
        self._cache: dict[_FmhaCacheKey, Fmha] = {}
        self._cache_inputs: dict[_FmhaCacheKey, _FmhaCacheInputs] = {}
        # Environment configuration is fixed for the manager lifetime so
        # normal forwarding never pays for an environment lookup.
        self._cache_sanity_check_enabled = _is_fmha_cache_sanity_check_enabled()
        self.fmha_libs: list[Fmha] = []
        for fmha_cls in get_enabled_fmha_lib_classes():
            if fmha_cls.is_available(attn):
                self.fmha_libs.append(fmha_cls(attn))

    def _make_cache_key(
        self,
        q: torch.Tensor,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> _FmhaCacheKey:
        """Build the dynamic FMHA cache key for one attention instance.

        FMHA cache inputs not represented here must remain invariants for
        the manager lifetime. Constructing a new manager starts a new cache.

        Batch size and generation Q length use ceiling grid buckets based on
        TRTLLM-Gen JIT warmup candidates. Each candidate and its predecessor
        are separate grid points so a support boundary at the candidate cannot
        alias with the interval below it. Generation Q lengths are uniform in
        the executor, including padded speculative-decoding batches.

        The batch fields describe the complete source scheduler batch, even
        when ``q`` contains only one compacted phase. Ordinary attention
        instances use mixed inputs, while MLA instances use those compacted
        phase calls, so the zero Q length still distinguishes a context-only
        subcall from a generation-active subcall.
        """
        attention_input_type = forward_args.attention_input_type
        output_dtype = forward_args.output.dtype if forward_args.output is not None else None
        output_sf_dtype = (
            forward_args.output_sf.dtype if forward_args.output_sf is not None else None
        )
        # Explicit mask data has the same FMHA support constraints as a
        # custom mask enum, even when the accompanying enum remains causal.
        attention_mask_type = (
            AttentionMaskType.custom_mask
            if (
                forward_args.attention_mask == CustomAttentionMask.CUSTOM
                or forward_args.attention_mask_data is not None
            )
            else AttentionMaskType(forward_args.mask_type)
        )

        context_batch_size = _normalize_fmha_cache_grid_value(
            metadata.num_contexts, _FMHA_CACHE_BATCH_SIZE_GRID
        )
        generation_batch_size = _normalize_fmha_cache_grid_value(
            metadata.num_generations, _FMHA_CACHE_BATCH_SIZE_GRID
        )
        generation_seq_len_q = 0
        if attention_input_type != AttentionInputType.context_only and metadata.num_generations > 0:
            generation_num_tokens = q.shape[0]
            if attention_input_type == AttentionInputType.mixed:
                generation_num_tokens -= metadata.num_ctx_tokens
            generation_seq_len_q = generation_num_tokens // metadata.num_generations
            generation_seq_len_q = _normalize_fmha_cache_grid_value(
                generation_seq_len_q, _FMHA_CACHE_SEQ_LEN_Q_GRID
            )

        return _FmhaCacheKey(
            context_batch_size=context_batch_size,
            generation_batch_size=generation_batch_size,
            generation_seq_len_q=generation_seq_len_q,
            attention_mask_type=attention_mask_type,
            use_spec_decoding=metadata.use_spec_decoding,
            is_cuda_graph=metadata.is_cuda_graph,
            is_fused_qkv=forward_args.is_fused_qkv,
            update_kv_cache=forward_args.update_kv_cache,
            output_dtype=output_dtype,
            output_sf_dtype=output_sf_dtype,
        )

    def select(
        self,
        attn: TrtllmAttention,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Fmha | None:
        """Select an FMHA library for one forward request."""
        if not _is_fmha_cache_enabled():
            return self._select_uncached(attn, q, k, v, metadata, forward_args)

        cache_key = self._make_cache_key(q, metadata, forward_args)
        fmha = self._cache.get(cache_key)
        if fmha is not None:
            if self._cache_sanity_check_enabled:
                uncached_fmha = self._select_uncached(attn, q, k, v, metadata, forward_args)
                if not _fmha_cache_values_match(fmha, uncached_fmha):
                    uncached_inputs = _snapshot_fmha_cache_inputs(q, k, v, metadata, forward_args)
                    input_diff = _format_fmha_cache_input_diff(
                        self._cache_inputs.get(cache_key), uncached_inputs
                    )
                    message = (
                        "FMHA cache sanity check failed for "
                        f"key={cache_key}: "
                        f"cached={_describe_fmha_cache_value(fmha)}, "
                        "uncached="
                        f"{_describe_fmha_cache_value(uncached_fmha)}.\n"
                        f"Forward input differences:\n{input_diff}"
                    )
                    logger.error(message)
                    raise RuntimeError(message)
                if cache_key not in self._cache_inputs:
                    self._cache_inputs[cache_key] = _snapshot_fmha_cache_inputs(
                        q, k, v, metadata, forward_args
                    )
            return fmha

        fmha = self._select_uncached(attn, q, k, v, metadata, forward_args)
        if fmha is None:
            return None
        self._cache[cache_key] = fmha
        if self._cache_sanity_check_enabled:
            self._cache_inputs[cache_key] = _snapshot_fmha_cache_inputs(
                q, k, v, metadata, forward_args
            )
        return fmha

    def _select_uncached(
        self,
        attn: TrtllmAttention,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Fmha | None:
        if attn.is_mla_enable:
            return self._select_mla(q, k, v, metadata, forward_args)
        return self._select_non_mla(attn, q, k, v, metadata, forward_args)

    def _select_non_mla(
        self,
        attn: TrtllmAttention,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Fmha | None:
        has_context = metadata.num_contexts > 0
        has_generation = metadata.num_generations > 0
        if not has_context and not has_generation:
            return None

        context_fmha = None
        generation_fmha = None
        for fmha in self.fmha_libs:
            if fmha.is_supported(q, k, v, metadata, forward_args):
                return fmha

            if not isinstance(fmha, PhasedFmha):
                continue

            if has_context and context_fmha is None:
                if fmha.is_supported(
                    q,
                    k,
                    v,
                    metadata,
                    forward_args,
                    phase=FmhaPhase.CONTEXT,
                ):
                    context_fmha = fmha
            if has_generation and generation_fmha is None:
                if fmha.is_supported(
                    q,
                    k,
                    v,
                    metadata,
                    forward_args,
                    phase=FmhaPhase.GENERATION,
                ):
                    generation_fmha = fmha

            if has_context and context_fmha is None:
                continue
            if has_generation and generation_fmha is None:
                continue
            if context_fmha is None:
                return generation_fmha
            if generation_fmha is None:
                return context_fmha
            if context_fmha is generation_fmha:
                continue

            combined_fmha = CombinedFmha(attn)
            combined_fmha.set_fmha_impls(context_fmha, generation_fmha)
            return combined_fmha
        return None

    def _select_mla(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> Fmha | None:
        if forward_args.attention_input_type == AttentionInputType.context_only:
            phase = FmhaPhase.CONTEXT
        elif forward_args.attention_input_type == AttentionInputType.generation_only:
            phase = FmhaPhase.GENERATION
        else:
            return None

        for fmha in self.fmha_libs:
            if isinstance(fmha, PhasedFmha):
                supported = fmha.is_supported(
                    q,
                    k,
                    v,
                    metadata,
                    forward_args,
                    phase=phase,
                )
            else:
                supported = fmha.is_supported(q, k, v, metadata, forward_args)
            if supported:
                return fmha
        return None


__all__ = ["FmhaManager"]
