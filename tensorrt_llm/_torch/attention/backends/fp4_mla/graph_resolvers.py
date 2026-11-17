# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the decode CUDA graphs of an FP4 MLA model are keyed by, besides their shape.

The decoder graph runner asks this module once, at capture, when the KV cache
exists. Both resolvers return None for a decode that needs no extra key, which
is every decode without an FP4 MLA KV cache and, by default, most with one.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from . import config as _config
from . import fp4_mla_dispatch, packed_plan

if TYPE_CHECKING:
    from torch import nn

    from tensorrt_llm.mapping import Mapping

# Regime ceilings of one graph shape: (batch size, draft length) -> KV lengths in tokens.
KvRegimeResolver = Callable[[int, int], tuple[int, ...]]
# Split decisions of one graph shape, and the decision of one step from its rows' KV lengths.
PackedSplitVariants = Callable[[int, int], tuple[bool, ...]]
PackedSplitDecision = Callable[[int, int, list[int]], bool]


def fp4_mla_4x_unavailable() -> str | None:
    """Why the current CUDA device cannot host the FP4 MLA 4x kernel; None when it can."""
    from .fp4_mla_cutedsl_4x_host import device_unsupported_reason

    return device_unsupported_reason(torch.device("cuda", torch.cuda.current_device()))


def fp4_mla_decode_heads(model: nn.Module) -> int | None:
    """Attention heads this rank's MLA layers decode with; None without MLA layers or with unequal ones."""
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
) -> KvRegimeResolver | None:
    """KV regimes the decode graphs are keyed by, per graph shape; None for every decode but FP4 MLA 4x.

    The decode dispatcher (``fp4_mla_dispatch``) then decides on the regime
    ceiling, the step's longest live KV rounded up, instead of the page-table
    capacity. The regimes of a shape are those its 2x | 4x | two-stream
    decision can tell apart (``kv_regimes_for_batch``), one for most shapes.
    None as well where every step decodes on the 2x kernel whatever its KV:
    with Helix, and on a device that cannot host the 4x kernel. The graph
    runner resolves every decode step, so each shape is computed once: replay
    keeps the ceilings its graphs were captured for.
    """
    # Only an FP4 MLA KV cache gets regimes (the predicate under which the
    # attention metadata carries an fp4_mla_state); every other model returns
    # here, before any FP4 MLA switch is read or the device probed.
    if not _config.has_fp4_mla_cache_layout(kv_cache_manager):
        return None
    # Environment gates next: the default (4x off) resolves nothing. The
    # dispatcher runs only for the packed-Q cutedsl decode and keeps Helix on
    # the 2x kernel, so regimes would only multiply those graphs.
    if not (_config._fp4_mla_cutedsl_4x_enabled() and _config._fp4_mla_packed_q_enabled()):
        return None
    if mapping.has_cp_helix():
        return None
    num_heads = fp4_mla_decode_heads(model)
    if num_heads is None:
        return None
    # A device that cannot host the 4x kernel decodes every step on 2x, as
    # Helix does: no regimes (the host logs the reason once per device).
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
) -> tuple[PackedSplitVariants, PackedSplitDecision] | None:
    """Packed split decisions the decode graphs are keyed by; None for every decode but the FP4 MLA packed one.

    The packed decode launches its KV split budget only for steps whose
    balanced plan splits a request (``TRTLLM_FP4_MLA_SPLIT_ON_DEMAND``, default
    on). A shape that fills two thirds of a 2-CTA wave captures an unsplit and
    a split graph, a smaller one the split graph alone
    (``packed_plan.packed_split_variants``); each step replays the graph of its
    decision (``packed_plan.packed_split_needed``), agreed across attention-DP
    ranks by the graph runner.

    Returns the decisions of a graph shape and the decision of a step.
    """
    if not _config.has_fp4_mla_cache_layout(kv_cache_manager):
        return None
    # The decision only exists for the packed decode outside Helix (which
    # keeps its split budget) and without a fixed split override. The
    # environment and topology come first, then the model; the packed-Q
    # check, which probes the device, last.
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
            # No packed decode for this query width: one family.
            return (True,)
        return packed_plan.packed_split_variants(int(batch_size), m_tiles=tiles, sm_count=sm_count)

    def decide(batch_size: int, draft_len: int, kv_lens: list[int]) -> bool:
        del batch_size  # the step's rows are kv_lens
        tiles = m_tiles(draft_len)
        if tiles <= 0:
            return True
        return packed_plan.packed_split_needed(
            kv_lens, query_len=int(draft_len) + 1, m_tiles=tiles, sm_count=sm_count
        )

    return variants, decide
