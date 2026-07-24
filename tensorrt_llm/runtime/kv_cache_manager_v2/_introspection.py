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

import sys
from typing import Any


def _cpp_introspection_module() -> Any | None:
    package = sys.modules.get(__package__)
    if package is None:
        return None
    return getattr(package, "_cpp_introspection", None)


def active_page_stats(kv_cache: Any) -> tuple[list[int], list[int]]:
    """Return active pages and unscheduled evictable active pages by cache level."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        counts, unscheduled_evictable = cpp_introspection.active_page_stats(kv_cache)
        return list(counts), list(unscheduled_evictable)

    storage = kv_cache.manager._storage
    counts = [0] * storage.num_cache_levels
    unscheduled_evictable = [0] * storage.num_cache_levels
    for ordinal, beam_idx, lc_idx in kv_cache._active_pages():
        block_page = kv_cache._page(ordinal, beam_idx, lc_idx)
        if block_page is None:
            continue

        page = block_page.page
        level = page.cache_level
        counts[level] += 1
        if storage.is_evictable(page) and not page.scheduled_for_eviction:
            unscheduled_evictable[level] += 1
    return counts, unscheduled_evictable


def all_tree_pages_droppable(manager: Any) -> bool:
    """Return whether every page reachable from the radix tree is droppable."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return bool(cpp_introspection.all_tree_pages_droppable(manager))

    from ._block_radix_tree import traverse_post_order
    from ._common import PageStatus
    from ._utils import unwrap_rawref

    for root_block in manager._radix_tree.next.values():
        for block0 in root_block.next.values():
            for block in traverse_post_order(block0):
                for page in block.storage:
                    if page is not None and unwrap_rawref(page).status != PageStatus.DROPPABLE:
                        return False
    return True


def is_commit_allowed(kv_cache: Any) -> bool:
    """Return whether the KV cache still allows token commits."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return bool(cpp_introspection.is_commit_allowed(kv_cache))
    return kv_cache._commit_state == kv_cache.CommitState.ALLOWED


def current_gpu_ratio(manager: Any) -> list[float]:
    """Return the current GPU pool-group ratio list."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return list(cpp_introspection.current_gpu_ratio(manager))
    return list(manager._current_gpu_ratio)


def set_num_sampled_kv_caches(manager: Any, value: int) -> None:
    """Set the sampled-KV-cache counter that gates auto-tuner rebalancing."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        cpp_introspection.set_num_sampled_kv_caches(manager, value)
        return
    manager._num_sampled_kv_caches = value


def set_last_adjustment_time(manager: Any, value: float) -> None:
    """Set the last pool-rebalance timestamp that gates the auto-tuner cooldown."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        cpp_introspection.set_last_adjustment_time(manager, value)
        return
    manager._last_adjustment_time = value


def set_target_ratio_list_gpu(manager: Any, ratios: list[float]) -> None:
    """Override the target GPU pool-group ratio list (drives the next rebalance)."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        cpp_introspection.set_target_ratio_list_gpu(manager, list(ratios))
        return
    manager._target_ratio_list_gpu = list(ratios)


def force_rebalance_precondition(manager: Any, skew: float = 2.0) -> None:
    """Force the V2 auto-tuner to do real pool-resize work on the next rebalance.

    Bypasses the sample-count / cooldown gates and perturbs the target GPU
    ratio so it differs from the current ratio by more than the auto-tuner's
    adjustment threshold. Requires a model with >=2 pool groups (e.g. a VSWA
    model) and raises ``ValueError`` otherwise, so a future model change can't
    silently turn a dependent test into a no-op. Backend-agnostic white-box
    hook intended for accuracy tests, not production code.
    """
    current = current_gpu_ratio(manager)
    if len(current) < 2:
        raise ValueError(
            f"force_rebalance_precondition requires >=2 pool groups; got {len(current)}. "
            "Check that VSWA is actually configured for this model."
        )
    set_num_sampled_kv_caches(manager, 2001)
    set_last_adjustment_time(manager, 0.0)
    skewed = [current[0] * skew] + current[1:]
    total = sum(skewed)
    set_target_ratio_list_gpu(manager, [x / total for x in skewed])


def storage_statistics(manager: Any, cache_level: int = 0) -> list[Any]:
    """Return storage statistics by pool group for a cache level."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return list(cpp_introspection.storage_statistics(manager, cache_level))
    return list(manager._storage.get_statistics(cache_level))


def storage_utilization(manager: Any, cache_level: int = 0) -> list[float]:
    """Return storage utilization by pool group for a cache level."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return list(cpp_introspection.storage_utilization(manager, cache_level))
    return list(manager._storage.get_utilization(cache_level))


def grains_for_slots(num_slots: int, slot_size_list: list[int], granularity: int) -> int:
    """Return the grain count required for a pool group slot count."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return int(cpp_introspection.grains_for_slots(num_slots, slot_size_list, granularity))

    from ._storage._core import CacheLevelStorage

    return int(CacheLevelStorage._grains_for_slots(num_slots, slot_size_list, granularity))


def grains_to_slots(pg_grains: int, slot_size_list: list[int], granularity: int) -> tuple[int, int]:
    """Return (slot count, consumed grains) for a pool group grain budget."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        slots, used = cpp_introspection.grains_to_slots(pg_grains, slot_size_list, granularity)
        return int(slots), int(used)

    from ._storage._core import CacheLevelStorage

    slots, used = CacheLevelStorage._grains_to_slots(pg_grains, slot_size_list, granularity)
    return int(slots), int(used)


def ratio_to_slot_count_list(
    quota: int,
    slot_size_lists: list[list[int]],
    ratio_list: list[float],
    granularity: int,
    min_slots: list[int],
) -> list[int]:
    """Return slot counts by pool group for a quota and ratio list."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return list(
            cpp_introspection.ratio_to_slot_count_list(
                quota, slot_size_lists, ratio_list, granularity, min_slots
            )
        )

    from ._storage._core import CacheLevelStorage

    return list(
        CacheLevelStorage.ratio_to_slot_count_list(
            quota, slot_size_lists, ratio_list, granularity, min_slots
        )
    )


def attention_life_cycle_ids(manager: Any) -> list[int]:
    """Return the lifecycle ids of all attention lifecycles, in order."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return list(cpp_introspection.attention_life_cycle_ids(manager))
    return [lc_id for lc_id, _ in manager._life_cycles.attention_life_cycles()]


def swa_life_cycle_ids(manager: Any) -> list[int]:
    """Return the lifecycle ids of attention lifecycles that use a sliding window."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return list(cpp_introspection.swa_life_cycle_ids(manager))
    return [
        lc_id
        for lc_id, lc in manager._life_cycles.attention_life_cycles()
        if lc.window_size is not None
    ]


def ssm_life_cycle_id(manager: Any) -> int | None:
    """Return the SSM lifecycle id, or None if there is no SSM lifecycle."""
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        return cpp_introspection.ssm_life_cycle_id(manager)
    return manager._life_cycles.ssm_life_cycle_id


def reuse_match_pages(
    manager: Any,
    reuse_scope: Any,
    tokens: Any,
    lc_id: int,
    enable_partial: bool = False,
) -> tuple[int, list[tuple[int, int | None] | None]]:
    """Match ``tokens`` against the radix tree and report reusable pages per block.

    Returns ``(num_tokens, pages)`` where ``pages[i]`` is ``None`` when block ``i``
    holds no page for lifecycle ``lc_id``, otherwise ``(slot_id, num_tokens_in_block)``
    with ``num_tokens_in_block`` set only for SSM pages (``None`` for attention pages).
    """
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is not None:
        num_tokens, raw_pages = cpp_introspection.reuse_match_pages(
            manager, reuse_scope, list(tokens), lc_id, enable_partial
        )
        pages: list[tuple[int, int | None] | None] = []
        for entry in raw_pages:
            if entry is None:
                pages.append(None)
            else:
                slot_id, num_tokens_in_block = entry
                pages.append((slot_id, None if num_tokens_in_block < 0 else num_tokens_in_block))
        return num_tokens, pages

    from ._utils import unwrap_rawref

    match = manager._radix_tree.match(reuse_scope, list(tokens), enable_partial)
    py_pages: list[tuple[int, int | None] | None] = []
    for block in match.blocks:
        ref = block.storage[lc_id]
        if ref is None:
            py_pages.append(None)
        else:
            page = unwrap_rawref(ref)
            py_pages.append((page.slot_id, getattr(page, "num_tokens_in_block", None)))
    return match.num_tokens, py_pages
