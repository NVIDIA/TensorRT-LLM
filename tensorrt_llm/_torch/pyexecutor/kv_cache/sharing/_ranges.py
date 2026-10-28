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
"""Retained and read-window block ranges, and the history required for resumption."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _manager

if TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2
    from ._layout import ManagerLayout


def _formula_stale_range(
    window: int, sink_blocks: int, tokens_per_block: int, history: int
) -> tuple[int, int]:
    """The runtime's stale range of a ``window`` with ``sink_blocks`` at ``history``, read at most
    at ``_MAX_TOKENS`` less one block as the manager reads it, in 32-bit integer division: covered
    sinks, then the blocks before the window ending at the history's next token."""
    history = min(int(history), _manager._MAX_TOKENS - tokens_per_block)
    start = min(-(-history // tokens_per_block), sink_blocks)
    behind = history + 1 - window
    window_start = behind // tokens_per_block if behind >= 0 else -(-behind // tokens_per_block)
    return start, max(start, window_start)


def _stale_range(
    manager: KVCacheManagerV2, layout: ManagerLayout, lg: int, history: int
) -> tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind layer group ``lg``'s window at ``history``: by the
    window formula where the layout found the manager follows it, else as the manager answers."""
    window = layout.windows[lg]
    if window is None:
        return 0, 0
    if getattr(layout, "stale_ranges_by_formula", False):
        return _formula_stale_range(
            window, layout.sink_blocks[lg], layout.tokens_per_block, history
        )
    return _manager.stale_range(manager, lg, history)


def _probed_histories(window: int, sink_blocks: int, tokens_per_block: int) -> list[int]:
    """Histories at which the window formula's terms change, or meet the manager's bounds."""
    tpb = tokens_per_block
    probes = set(range(3 * tpb + 2))
    for k in range(8):
        for d in range(-2, 3):
            probes.add(window + k * tpb + d)
            probes.add((k + 1) * tpb + window - 2 + d)
            probes.add((sink_blocks + k) * tpb + d)
    longest = _manager._MAX_TOKENS - tpb
    probes.update((10007, 1 << 20, longest - 1, longest, longest + 1, _manager._MAX_TOKENS))
    return sorted(h for h in probes if 0 <= h <= _manager._MAX_TOKENS)


def _stale_ranges_follow_formula(manager: KVCacheManagerV2, layout: ManagerLayout) -> bool:
    """Whether the manager's stale range of every windowed layer group equals the window formula at
    every probed history, without Helix."""
    if getattr(layout, "cp", (0, 1))[1] != 1:
        return False
    tpb = int(layout.tokens_per_block)
    for lg, window in enumerate(layout.windows):
        if window is None:
            continue
        sinks = int(layout.sink_blocks[lg])
        for history in _probed_histories(int(window), sinks, tpb):
            formula = _formula_stale_range(int(window), sinks, tpb, history)
            if tuple(_manager.stale_range(manager, lg, history)) != formula:
                return False
    return True


def _needed_ranges(
    manager: KVCacheManagerV2,
    layout: ManagerLayout,
    lg: int,
    start_block: int,
    end_block: int,
    history: int,
) -> list[tuple[int, int]]:
    """Nonempty retained ordinal ranges within ``[start_block, end_block)`` at ``history``.
    Full attention keeps the whole range; windows keep sinks and their retained suffix."""
    stale_beg, stale_end = _stale_range(manager, layout, lg, history)
    runs = ((start_block, min(end_block, stale_beg)), (max(start_block, stale_end), end_block))
    return [(beg, end) for beg, end in runs if end > beg]


def _read_slack(layout: ManagerLayout, lg: int) -> int:
    """Tokens by which layer group ``lg`` keeps a window wider than the one its attention reads."""
    read_windows = getattr(layout, "read_windows", ())
    if not read_windows or layout.windows[lg] is None:
        return 0
    return int(layout.windows[lg]) - int(read_windows[lg])


def _read_stale_range(
    manager: KVCacheManagerV2, layout: ManagerLayout, lg: int, history: int
) -> tuple[int, int]:
    """Unread ordinals at ``history``, excluding sinks, using the read-window slack."""
    return _stale_range(manager, layout, lg, history + _read_slack(layout, lg))


def _last_history_keeping(
    manager: KVCacheManagerV2, layout: ManagerLayout, lg: int, block: int
) -> int:
    """The largest history at which layer group ``lg``'s window still keeps ``block``, a block past
    its sinks: the stale range's end, which grows with the history, stays at or below it."""
    tpb = int(layout.tokens_per_block)
    if getattr(layout, "stale_ranges_by_formula", False) and block >= layout.sink_blocks[lg]:
        # Past the sinks the stale end is the window's start, (history + 1 - window) // tpb.
        last = (block + 1) * tpb + int(layout.windows[lg]) - 2
        return _manager._MAX_TOKENS if last >= _manager._MAX_TOKENS - tpb else last
    low = block * tpb  # inside the window: no history this short leaves the block behind
    high = low + tpb + int(layout.windows[lg])
    while _stale_range(manager, layout, lg, high)[1] <= block:
        if high >= _manager._MAX_TOKENS:
            return _manager._MAX_TOKENS
        high = min(2 * high, _manager._MAX_TOKENS)
    while high - low > 1:
        middle = (low + high) // 2
        if _stale_range(manager, layout, lg, middle)[1] <= block:
            low = middle
        else:
            high = middle
    return low


def _last_history_reading(
    manager: KVCacheManagerV2, layout: ManagerLayout, lg: int, block: int
) -> int:
    """The largest history at which layer group ``lg``'s attention still reads ``block``: the last
    the kept window keeps it, less the slack between the windows."""
    return _last_history_keeping(manager, layout, lg, block) - _read_slack(layout, lg)


# TODO: a draft pool also gives up a resume below its history whose chunk would reach it, as
# every unchunked prefill's does.
def _floor_follows_history(manager: KVCacheManagerV2) -> bool:
    """Whether a request may resume only at or past its history: a chunk resumed below it may end
    there (the context update raises) or leave a draft pool's capacity there (its resize raises)."""
    return _manager.context_moves_history(manager) or _manager.resize_ends_at_chunk(manager)
