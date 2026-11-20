# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the decode CUDA graphs of an FP4 MLA model are keyed by, besides their shape.

The decoder graph runner asks the attention backend once, at capture, when the
KV cache exists (``AttentionBackend.decode_graph_variants``), and the TRTLLM
backend answers with ``resolve_fp4_mla_graph_variants``. It gets a graph
variant (``interface.GraphVariant``) for each thing the FP4 MLA decode decides
per step and a graph has to freeze: the KV regime the dispatcher chooses its
kernel on, and whether the packed decode launches its KV split budget. A
decode that needs neither gets none, which is every decode without an FP4 MLA
KV cache and, by default, most with one.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from ..interface import GraphVariant
from . import config as _config
from . import fp4_mla_dispatch, packed_plan
from .state import KV_REGIME_VARIANT, PACKED_SPLIT_VARIANT

if TYPE_CHECKING:
    from torch import nn

    from tensorrt_llm.mapping import Mapping

# Regime ceilings of one graph shape: (batch size, draft length) -> KV lengths in tokens.
KvRegimeResolver = Callable[[int, int], tuple[int, ...]]


class PackedSplitDecisions(NamedTuple):
    """Split decisions of the packed decode that decode graphs are keyed by.

    ``variants(batch_size, draft_len)`` lists the decisions the graphs of a
    shape are captured for (``packed_plan.packed_split_variants``);
    ``decide(batch_size, draft_len, kv_lens)`` is a step's decision from its
    per-request KV lengths (``packed_plan.packed_split_needed``).
    """

    variants: Callable[[int, int], tuple[bool, ...]]
    decide: Callable[[int, int, list[int]], bool]


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
) -> PackedSplitDecisions | None:
    """Packed split decisions the decode graphs are keyed by; None for every decode but the FP4 MLA packed one.

    The packed decode launches its KV split budget only for steps whose
    balanced plan splits a request (``TRTLLM_FP4_MLA_SPLIT_ON_DEMAND``, default
    on). A shape that fills two thirds of a 2-CTA wave captures an unsplit and
    a split graph, a smaller one the split graph alone
    (``packed_plan.packed_split_variants``); each step replays the graph of its
    decision (``packed_plan.packed_split_needed``), agreed across attention-DP
    ranks by the graph runner.
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

    return PackedSplitDecisions(variants=variants, decide=decide)


def select_kv_regime_ceiling(max_kv_len: int, ceilings: Sequence[int]) -> int:
    """Regime of a step whose longest live KV is ``max_kv_len``: the smallest ceiling covering it.

    Shorter steps share the first regime; a step beyond the list (never, when
    ``ceilings`` ends with the capacity) gets the last one, the graph of the
    capacity.
    """
    for ceiling in ceilings:
        if max_kv_len <= ceiling:
            return int(ceiling)
    return int(ceilings[-1])


def kv_regime_capture_passes(
    warmup_lengths: Sequence[int], ceilings: tuple[int, ...]
) -> list[tuple[int, int]]:
    """(warm-up length, KV regime) of every generation graph family a step can key, largest first.

    For each warm-up length (the engine's maximum, plus the sparse short length when the
    sparse attention keeps short and long graph families apart) a step can land in every
    regime whose ceiling lies below that length and in the regime covering the length
    itself. The warm-up request's longest row fills the regime, so the dispatcher decision
    a graph captures is the one of its regime.
    """
    passes: set[tuple[int, int]] = set()
    for max_seq_len in warmup_lengths:
        passes.update((ceiling, ceiling) for ceiling in ceilings if ceiling < max_seq_len)
        passes.add((max_seq_len, select_kv_regime_ceiling(max_seq_len, ceilings)))
    return sorted(passes, reverse=True)


def _replayable_kv_regime(regime: int, captured: Sequence[int]) -> int:
    """Round the KV regime up to a captured one.

    A graph of a larger regime is valid for the step: the kernels are planned
    for the page-table capacity and only the dispatcher's decision is baked
    in. A regime whose capture was skipped replays the next one up, or the
    largest captured when none covers it.
    """
    if not captured:
        return regime
    covering = [ceiling for ceiling in captured if ceiling >= regime]
    return min(covering) if covering else max(captured)


def _replayable_packed_split(split: int, captured: Sequence[int]) -> int:
    """Replay the split graph when the step's unsplit graph was not captured.

    A split graph is valid for any step: its plan keeps every request whole
    when the step needs no split, at the cost of the empty slots. Shapes with
    a single (split) variant and skipped captures fall back to it.
    """
    return split if split or 0 in captured else 1


def kv_regime_variant(ceilings: KvRegimeResolver) -> GraphVariant:
    """The KV regime as a graph variant: the value is the regime's ceiling in tokens."""

    def of_step(batch_size: int, draft_len: int, kv_lens: list[int]) -> int:
        return select_kv_regime_ceiling(max(kv_lens, default=0), ceilings(batch_size, draft_len))

    return GraphVariant(
        name=KV_REGIME_VARIANT,
        values=ceilings,
        of_step=of_step,
        replayable=_replayable_kv_regime,
        capture_passes=kv_regime_capture_passes,
    )


def packed_split_variant(decisions: PackedSplitDecisions) -> GraphVariant:
    """The packed split decision as a graph variant: 1 launches the split budget, 0 one task per request."""

    def values(batch_size: int, draft_len: int) -> tuple[int, ...]:
        return tuple(sorted(int(split) for split in decisions.variants(batch_size, draft_len)))

    def of_step(batch_size: int, draft_len: int, kv_lens: list[int]) -> int:
        return int(decisions.decide(batch_size, draft_len, kv_lens))

    return GraphVariant(
        name=PACKED_SPLIT_VARIANT,
        values=values,
        of_step=of_step,
        replayable=_replayable_packed_split,
    )


def resolve_fp4_mla_graph_variants(
    *,
    kv_cache_manager: Any,
    model: nn.Module,
    mapping: Mapping,
    sm_count_fn: Callable[[], int],
    four_x_unavailable_fn: Callable[[], str | None] = fp4_mla_4x_unavailable,
) -> list[GraphVariant]:
    """The graph variants the decode graphs of this model are keyed by; none for most decodes.

    The KV regime comes first: its capture passes are tied to the warm-up
    lengths, and the packed split decision is settled before it when a step
    replays a graph of another value.
    """
    variants = []
    regimes = resolve_fp4_mla_kv_regimes(
        kv_cache_manager=kv_cache_manager,
        model=model,
        mapping=mapping,
        sm_count_fn=sm_count_fn,
        four_x_unavailable_fn=four_x_unavailable_fn,
    )
    if regimes is not None:
        variants.append(kv_regime_variant(regimes))
    decisions = resolve_fp4_mla_packed_split(
        kv_cache_manager=kv_cache_manager, model=model, mapping=mapping, sm_count_fn=sm_count_fn
    )
    if decisions is not None:
        variants.append(packed_split_variant(decisions))
    return variants
