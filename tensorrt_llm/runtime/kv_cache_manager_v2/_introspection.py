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

"""Internal KVCacheManagerV2 surface: no stability promise, may change with the C++ side.

Everything here forwards to the native ``_introspection`` submodule of the bindings. The
indirection exists so callers import one Python path rather than reaching into
``tensorrt_llm.bindings`` directly, and so return values are normalised to plain Python
containers. Most hooks are white-box helpers for tests and accuracy harnesses, but a few
back production internals -- the stats report and the disaggregated bounce buffer -- so
treat this as internal API rather than test-only scaffolding.
"""

from __future__ import annotations

import sys
from typing import Any


def _cpp_introspection_module() -> Any | None:
    package = sys.modules.get(__package__)
    if package is None:
        return None
    return getattr(package, "_cpp_introspection", None)


def _cpp() -> Any:
    cpp_introspection = _cpp_introspection_module()
    if cpp_introspection is None:
        raise RuntimeError(
            "KVCacheManagerV2 introspection requires the native _introspection module, "
            "which is missing from this build of tensorrt_llm.bindings"
        )
    return cpp_introspection


#: CUDA virtual-memory primitives, forwarded rather than wrapped: they are classes the
#: caller constructs, so there is nothing to normalise. Used by the native disaggregated
#: bounce buffer to reserve one contiguous fabric region.
_FORWARDED_TYPES = ("PooledPhysMemAllocator", "VirtMem")


def __getattr__(name: str) -> Any:
    # Resolved on first access, not at import, so importing this module never depends on
    # the native submodule being present -- the same contract the hooks below follow.
    if name in _FORWARDED_TYPES:
        return getattr(_cpp(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def create_test_padding_cold_page_codec(cold_page_bytes_by_layer: dict[int, int]) -> Any:
    """Create the private native padding codec used by cold-tier end-to-end tests."""
    return _cpp().create_test_padding_cold_page_codec(cold_page_bytes_by_layer)


def make_test_block(
    manager: Any,
    tokens: Any,
    coverage_per_lc: list[int],
    parent: Any = None,
    reuse_scope: Any = None,
) -> Any:
    """Build a real block with real GPU pages at the requested lifecycle coverage."""
    return _cpp().make_test_block(manager, list(tokens), coverage_per_lc, parent, reuse_scope)


def close_test_block(block: Any) -> None:
    """Unlink and release a test block pages before its manager shuts down."""
    block.close()


def test_block_key(block: Any) -> bytes:
    """Return a test block real radix-tree key."""
    return bytes(_cpp().test_block_key(block))


def event_manager_add_stored_block(event_manager: Any, block: Any) -> None:
    """Derive and enqueue stored events from a real test block."""
    _cpp().event_manager_add_stored_block(event_manager, block)


def event_manager_add_stored_life_cycle(event_manager: Any, block: Any, life_cycle_id: int) -> None:
    """Derive and enqueue one lifecycle stored event from a real test block."""
    _cpp().event_manager_add_stored_life_cycle(event_manager, block, life_cycle_id)


def active_page_stats(kv_cache: Any) -> tuple[list[int], list[int]]:
    """Return active pages and unscheduled evictable active pages by cache level."""
    counts, unscheduled_evictable = _cpp().active_page_stats(kv_cache)
    return list(counts), list(unscheduled_evictable)


def committed_page_is_linked(kv_cache: Any, ordinal: int, lc_id: int) -> bool | None:
    """Whether the sequence's page at ``(ordinal, lc_id)`` still points at a tree block.

    ``None`` when the slot is empty or holds an uncommitted page. Test hook: in C++ the
    back-pointer is raw, so a page left pointing at a block that dies first is read after
    free, and freed-but-mapped memory reads back plausibly enough that only a sanitizer
    build catches the fault itself.
    """
    return _cpp().committed_page_is_linked(kv_cache, ordinal, lc_id)


def all_tree_pages_droppable(manager: Any) -> bool:
    """Return whether every page reachable from the radix tree is droppable."""
    return bool(_cpp().all_tree_pages_droppable(manager))


def is_commit_allowed(kv_cache: Any) -> bool:
    """Return whether the KV cache still allows token commits."""
    return bool(_cpp().is_commit_allowed(kv_cache))


def current_gpu_ratio(manager: Any) -> list[float]:
    """Return the current GPU pool-group ratio list."""
    return list(_cpp().current_gpu_ratio(manager))


def set_num_sampled_kv_caches(manager: Any, value: int) -> None:
    """Set the sampled-KV-cache counter that gates auto-tuner rebalancing."""
    _cpp().set_num_sampled_kv_caches(manager, value)


def set_last_adjustment_time(manager: Any, value: float) -> None:
    """Set the auto-tuner's last-adjustment timestamp."""
    _cpp().set_last_adjustment_time(manager, value)


def set_target_ratio_list_gpu(manager: Any, ratios: list[float]) -> None:
    """Set the auto-tuner's target GPU pool-group ratio list."""
    _cpp().set_target_ratio_list_gpu(manager, list(ratios))


def force_rebalance_precondition(manager: Any, skew: float = 2.0) -> None:
    """Force the V2 auto-tuner to do real pool-resize work on the next rebalance.

    Bypasses the sample-count / cooldown gates and perturbs the target GPU
    ratio so it differs from the current ratio by more than the auto-tuner's
    adjustment threshold. Requires a model with >=2 pool groups (e.g. a VSWA
    model) and raises ``ValueError`` otherwise, so a future model change can't
    silently turn a dependent test into a no-op. White-box hook intended for
    accuracy tests, not production code.
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
    return list(_cpp().storage_statistics(manager, cache_level))


def storage_utilization(manager: Any, cache_level: int = 0) -> list[float]:
    """Return storage utilization by pool group for a cache level."""
    return list(_cpp().storage_utilization(manager, cache_level))


def grains_for_slots(num_slots: int, slot_size_list: list[int], granularity: int) -> int:
    """Return the grain count required for a pool group slot count."""
    return int(_cpp().grains_for_slots(num_slots, slot_size_list, granularity))


def grains_to_slots(pg_grains: int, slot_size_list: list[int], granularity: int) -> tuple[int, int]:
    """Return (slot count, consumed grains) for a pool group grain budget."""
    slots, used = _cpp().grains_to_slots(pg_grains, slot_size_list, granularity)
    return int(slots), int(used)


def ratio_to_slot_count_list(
    quota: int,
    slot_size_lists: list[list[int]],
    ratio_list: list[float],
    granularity: int,
    min_slots: list[int],
) -> list[int]:
    """Return slot counts by pool group for a quota and ratio list."""
    return list(
        _cpp().ratio_to_slot_count_list(quota, slot_size_lists, ratio_list, granularity, min_slots)
    )


def attention_life_cycle_ids(manager: Any) -> list[int]:
    """Return the lifecycle ids of all attention lifecycles, in order."""
    return list(_cpp().attention_life_cycle_ids(manager))


def swa_life_cycle_ids(manager: Any) -> list[int]:
    """Return the lifecycle ids of attention lifecycles that use a sliding window."""
    return list(_cpp().swa_life_cycle_ids(manager))


def ssm_life_cycle_id(manager: Any) -> int | None:
    """Return the SSM lifecycle id, or None if there is no SSM lifecycle."""
    return _cpp().ssm_life_cycle_id(manager)


def reuse_match_pages(
    manager: Any,
    reuse_scope: Any,
    tokens: Any,
    lc_id: int,
    enable_partial: bool = False,
) -> tuple[int, list[tuple[int, int | None] | None]]:
    """Match ``tokens`` against the radix tree and report reusable pages per block.

    Returns ``(num_tokens, pages)`` where ``pages[i]`` is ``None`` when block ``i``
    holds no page for lifecycle ``lc_id``, otherwise ``(slot_id, num_tokens_in_block)``.
    See ``CommittedPage.num_tokens_in_block`` for how attention and SSM life cycles
    interpret the recorded token count.
    """
    num_tokens, raw_pages = _cpp().reuse_match_pages(
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


def reuse_match_planned_drop_counts(
    manager: Any,
    reuse_scope: Any,
    tokens: Any,
    lc_id: int,
    enable_partial: bool = False,
) -> tuple[int, list[int | None]]:
    """Match ``tokens`` and report each matched block's ``planned_drop_count`` for lifecycle ``lc_id``.

    Returns ``(num_tokens, counts)`` where ``counts[i]`` is ``None`` when block ``i`` holds no page
    for lifecycle ``lc_id``, otherwise the matched page's ``planned_drop_count``.
    """
    return _cpp().reuse_match_planned_drop_counts(
        manager, reuse_scope, list(tokens), lc_id, enable_partial
    )


def pool_group_index(manager: Any, lc_id: int, cache_level: int = 0) -> int:
    """Return a lifecycle storage pool-group index at ``cache_level``."""
    return _cpp().pool_group_index(manager, lc_id, cache_level)


def life_cycle_pool_group_indices(manager: Any, cache_level: int = 0) -> list[int]:
    """Return the pool-group index of every lifecycle at ``cache_level``, indexed by lifecycle ID.

    Cold levels group lifecycles by encoded cold-page size, so their pool-group indices and counts are
    unrelated to the hot ones. Callers that aggregate level-specific data must translate through this
    mapping rather than reusing hot pool-group indices.
    """
    return list(_cpp().life_cycle_pool_group_indices(manager, cache_level))


def compute_slots_for_batch(
    manager: Any,
    batch: Any,
    tokens_per_block: int,
    swa_scratch_reuse: Any = None,
) -> list[int]:
    """Return the minimum per-pool-group slot counts to support ``batch``."""
    return list(_cpp().compute_slots_for_batch(manager, batch, tokens_per_block, swa_scratch_reuse))
