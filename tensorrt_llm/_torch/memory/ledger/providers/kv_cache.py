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
"""KV cache v2 provider.

The pool is allocated once and held for the process's life, so its physical
footprint is the *configured* capacity -- ``total`` slots -- and does not
shrink as blocks are freed. ``free`` and ``evictable`` describe how that
fixed allocation is being used, which is a logical split; subtracting them
from the physical figure would report memory the device has not taken back.

Only the three stored fields of ``StorageStatistics`` are read. The object
also exposes ``available()`` and ``unavailable()``, but those are computed
from the same three (``free + evictable`` and ``total - available()``,
``storageManager.h:84``), so treating them as independent terms turns a
conservation check into a tautology.
"""

from ..schema import Basis, Category, Entry, Placement, Source

__all__ = ["KvCacheProvider", "kv_pool_group_stats"]

GPU_LEVEL_FALLBACK = 0


def _slot_bytes(stats) -> int:
    """Bytes per block: every pool in the group contributes one slot."""
    return sum(int(size) for size in stats.slot_sizes)


def kv_pool_group_stats(manager) -> list[dict]:
    """Per-pool-group figures in bytes, derived only from stored fields.

    Returned rather than logged so the conservation checks and the ledger
    entries work from one reading; sampling twice would let the pool change
    underneath a comparison.
    """
    try:
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import GPU_LEVEL
    except ImportError:
        GPU_LEVEL = GPU_LEVEL_FALLBACK

    per_group = manager._get_storage_statistics(GPU_LEVEL)

    rows = []
    for pool_id, stats in enumerate(per_group):
        slot_bytes = _slot_bytes(stats)
        total = int(stats.total)
        free = int(stats.free)
        evictable = int(stats.evictable)
        rows.append(
            {
                "pool_group_id": pool_id,
                "slot_bytes": slot_bytes,
                "total_slots": total,
                "free_slots": free,
                "evictable_slots": evictable,
                # Physical: the whole pool is resident whether or not it is in use.
                "capacity_bytes": total * slot_bytes,
                # Logical: how that resident capacity is currently divided.
                "free_bytes": free * slot_bytes,
                "evictable_bytes": evictable * slot_bytes,
                "in_use_bytes": (total - free - evictable) * slot_bytes,
            }
        )
    return rows


class KvCacheProvider:
    """KV cache v2 pools.

    Takes the manager explicitly: it lives in a ``ResourceManager`` owned by
    the executor, with no process-wide accessor, so it cannot be discovered
    the way ``MnnvlMemory`` can. The capture site inside the executor has the
    reference and passes it in.
    """

    name = "kv_cache_v2"
    category = Category.KV_CACHE_POOL

    def __init__(self, manager=None) -> None:
        self._manager = manager

    def available(self) -> bool:
        return self._manager is not None and hasattr(self._manager, "_get_storage_statistics")

    def collect(self) -> list[Entry]:
        if not self.available():
            return []

        rows = kv_pool_group_stats(self._manager)
        entries = []

        for row in rows:
            entries.append(
                Entry(
                    category=Category.KV_CACHE_POOL,
                    size_bytes=row["capacity_bytes"],
                    source=Source.LEDGER,
                    placement=Placement.NON_TORCH,
                    basis=Basis.PHYSICAL_BACKING,
                    owner=f"kv_cache_v2.pool_group[{row['pool_group_id']}]",
                    note=(
                        f"{row['total_slots']} slots resident; "
                        f"{row['in_use_bytes']} B in use, "
                        f"{row['free_bytes']} B free, "
                        f"{row['evictable_bytes']} B evictable"
                    ),
                )
            )
            # The split is reported for diagnosis but never summed: these
            # bytes are already inside capacity_bytes above.
            entries.append(
                Entry(
                    category=Category.KV_CACHE_POOL,
                    size_bytes=row["in_use_bytes"],
                    source=Source.LEDGER,
                    placement=Placement.NON_TORCH,
                    basis=Basis.LOGICAL,
                    owner=f"kv_cache_v2.pool_group[{row['pool_group_id']}]",
                    note="in use within the resident pool",
                    counts_toward_total=False,
                )
            )

        return entries
