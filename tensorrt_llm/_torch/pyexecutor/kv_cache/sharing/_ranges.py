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
"""Which block ordinals of a layer group a history still reads under its window, and whether a
resume must stay at or past the history. Other modules call these through the module object
(``_ranges._stale_range(...)``), so each has one binding, which every caller reads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from . import _manager

if TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2
    from ._layout import ManagerLayout


def _stale_range(
    manager: KVCacheManagerV2, layout: ManagerLayout, lg: int, history: int
) -> tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind layer group ``lg``'s window at ``history``."""
    if layout.windows[lg] is None:
        return 0, 0
    return _manager.stale_range(manager, lg, history)


def _needed_ranges(
    manager: KVCacheManagerV2,
    layout: ManagerLayout,
    lg: int,
    start_block: int,
    end_block: int,
    history: int,
) -> list[tuple[int, int]]:
    """Ordinal ranges ``[beg, end)`` of ``lg`` in ``[start_block, end_block)`` that a history of
    ``history`` tokens still reads, in order and none empty: the whole range for full attention;
    the sinks and the window otherwise. Arithmetic only, so a range of any length costs nothing."""
    # TODO: DeepSeek-V4 widens every window by the draft length ``D`` under any speculative
    # decoding, so at 128 tokens per block and ``D >= 2`` a fetch asks for margin rows that a
    # publish from a history at least ``D - 1`` tokens past the fetch's end leaves out.
    stale_beg, stale_end = _stale_range(manager, layout, lg, history)
    runs = ((start_block, min(end_block, stale_beg)), (max(start_block, stale_end), end_block))
    return [(beg, end) for beg, end in runs if end > beg]


def _ordinals(runs: Sequence[tuple[int, int]]) -> np.ndarray:
    """``int64`` ordinals of ``runs``, in order."""
    if not runs:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(beg, end, dtype=np.int64) for beg, end in runs])


def _needed_ordinals(
    manager: KVCacheManagerV2,
    layout: ManagerLayout,
    lg: int,
    start_block: int,
    end_block: int,
    history: int,
) -> np.ndarray:
    """Ordinals of ``lg`` in ``[start_block, end_block)`` that a history of ``history`` tokens
    still reads: all of them for full attention; the sinks and the window otherwise."""
    return _ordinals(_needed_ranges(manager, layout, lg, start_block, end_block, history))


# TODO: a draft pool also gives up a resume below its history whose chunk would reach it, as
# every unchunked prefill's does.
def _floor_follows_history(manager: KVCacheManagerV2) -> bool:
    """Whether a request may resume only at or past its history: a chunk resumed below it may end
    below it, where the manager's context update raises, or leave a draft pool's capacity below it,
    where the pool's context resize raises."""
    return _manager.context_moves_history(manager) or _manager.resize_ends_at_chunk(manager)
