# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode CUDA graph regime and split-budget resolvers for FP4 MLA."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import torch

from . import config as _config
from . import fp4_mla_dispatch, packed_plan

if TYPE_CHECKING:
    from torch import nn

    from tensorrt_llm.mapping import Mapping

_ResolverT = TypeVar("_ResolverT")


def fp4_mla_4x_unavailable() -> str | None:
    """Return why the current CUDA device cannot host the FP4 MLA 4x kernel, or None if it can."""
    from .fp4_mla_cutedsl_4x_host import device_unsupported_reason

    return device_unsupported_reason(torch.device("cuda", torch.cuda.current_device()))


def fp4_mla_decode_heads(model: nn.Module) -> int | None:
    """Return the uniform TP-local MLA head count on this rank, or None without uniform MLA layers."""
    from tensorrt_llm._torch.attention.mla import MLA

    heads = {int(module.num_heads_tp) for module in model.modules() if isinstance(module, MLA)}
    return heads.pop() if len(heads) == 1 else None


def resolve_fp4_mla_kv_regimes(
    *,
    kv_cache_manager: Any,
    model: nn.Module,
    mapping: Mapping,
    sm_count_fn: Callable[[], int],
    four_x_unavailable_fn: Callable[[], str | None] = fp4_mla_4x_unavailable,
) -> Callable[[int, int], tuple[int, ...]] | None:
    """Build the per-shape KV-regime resolver for FP4 MLA 4x decode graphs, or None when inactive."""
    if not _config.has_fp4_mla_cache_layout(kv_cache_manager):
        return None
    if not (_config._fp4_mla_cutedsl_4x_enabled() and _config._fp4_mla_packed_q_enabled()):
        return None
    if mapping.has_cp_helix():
        return None
    num_heads = fp4_mla_decode_heads(model)
    if num_heads is None:
        return None
    if four_x_unavailable_fn() is not None:
        return None

    from .fp4_mla_cutedsl_4x_host import envelope_reason_static

    capacity = int(kv_cache_manager.max_blocks_per_seq) * _config.FP4_MLA_TOKENS_PER_BLOCK
    sm_count = sm_count_fn()

    @functools.cache
    def resolver(batch_size: int, draft_len: int) -> tuple[int, ...]:
        query_len = int(draft_len) + 1
        return fp4_mla_dispatch.kv_regimes_for_batch(
            num_requests=int(batch_size),
            query_len=query_len,
            num_heads=num_heads,
            rows_per_seq=_config._fp4_mla_packed_rows_per_seq(num_heads, query_len),
            kv_capacity=capacity,
            sm_count=sm_count,
            envelope_reason=envelope_reason_static,
        )

    return resolver


def resolve_fp4_mla_packed_split(
    *,
    kv_cache_manager: Any,
    model: nn.Module,
    mapping: Mapping,
    sm_count_fn: Callable[[], int],
    resolver_cls: Callable[..., _ResolverT],
) -> _ResolverT | None:
    """Build the packed-split resolver for FP4 MLA packed decode graphs, or None when inactive."""
    if not _config.has_fp4_mla_cache_layout(kv_cache_manager):
        return None
    if not _config._fp4_mla_split_on_demand_enabled():
        return None
    if _config._fp4_mla_split_kv_override() != "auto":
        return None
    if mapping.has_cp_helix():
        return None
    num_heads = fp4_mla_decode_heads(model)
    if num_heads is None:
        return None
    if not _config._fp4_mla_packed_q_enabled():
        return None
    sm_count = sm_count_fn()

    @functools.cache
    def m_tiles(draft_len: int) -> int:
        rows = _config._fp4_mla_packed_rows_per_seq(num_heads, int(draft_len) + 1)
        return rows // _config.FP4_MLA_SCALE_ROW_GROUP

    @functools.cache
    def variants(batch_size: int, draft_len: int) -> tuple[bool, ...]:
        tiles = m_tiles(draft_len)
        if tiles <= 0:
            return (True,)
        return packed_plan.packed_split_variants(int(batch_size), m_tiles=tiles, sm_count=sm_count)

    def decide(batch_size: int, draft_len: int, kv_lens: list[int]) -> bool:
        del batch_size
        tiles = m_tiles(draft_len)
        if tiles <= 0:
            return True
        return packed_plan.packed_split_needed(
            kv_lens, query_len=int(draft_len) + 1, m_tiles=tiles, sm_count=sm_count
        )

    return resolver_cls(variants=variants, decide=decide)
