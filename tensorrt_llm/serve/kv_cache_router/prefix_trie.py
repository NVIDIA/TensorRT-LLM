# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Block-hash prefix index mapping block-hash chains to the workers that hold them.

A KV-cache block hash folds in its parent chain (the worker computes it with
``BlockKeyHasher.hash(block_key, parent_hash)``), so every block-hash *value* is
globally unique to one position in one prefix path. That lets us index ownership
with a flat ``hash -> {worker_id}`` map instead of an explicit tree: a request's
ordered block-hash list is itself the prefix path, and longest-common-prefix per
worker falls out of walking the list while intersecting owner sets.

All operations are O(L) in the number of query / changed blocks, with no
recursive pruning. One index is kept per routing namespace by the router so
queries never cross groups. The class is named ``WorkerPrefixTrie`` for
continuity with the design doc; structurally it is a flat prefix index.
"""

from collections import defaultdict
from typing import Dict, Iterable, List

__all__ = ["WorkerPrefixTrie"]


class WorkerPrefixTrie:
    """Tracks which workers hold each KV-cache block and answers prefix queries."""

    def __init__(self) -> None:
        # block_hash -> set of worker_ids holding that block
        self._owners: Dict[int, set[str]] = defaultdict(set)
        # worker_id -> set of block hashes it holds (for O(blocks) eviction)
        self._worker_blocks: Dict[str, set[int]] = defaultdict(set)

    def add(self, worker_id: str, block_hashes: Iterable[int]) -> None:
        """Register *worker_id* as holding each of *block_hashes*. Idempotent."""
        owned = self._worker_blocks[worker_id]
        for h in block_hashes:
            self._owners[h].add(worker_id)
            owned.add(h)

    def remove(self, worker_id: str, block_hashes: Iterable[int]) -> None:
        """Drop *worker_id* from the given blocks (e.g. on a ``removed`` event)."""
        owned = self._worker_blocks.get(worker_id)
        if owned is None:
            return
        for h in block_hashes:
            self._discard(worker_id, h)
            owned.discard(h)
        if not owned:
            self._worker_blocks.pop(worker_id, None)

    def remove_worker(self, worker_id: str) -> None:
        """Drop *worker_id* entirely (full-snapshot replace or disconnect)."""
        owned = self._worker_blocks.pop(worker_id, None)
        if not owned:
            return
        for h in owned:
            self._discard(worker_id, h)

    def match(self, block_hashes: List[int]) -> Dict[str, int]:
        """Return ``worker_id -> number of consecutive prefix blocks matched``.

        Walks the query path; a worker's count is the depth at which it stops
        owning the running prefix. Workers that do not own the first block match
        zero blocks and are omitted.
        """
        result: Dict[str, int] = {}
        candidates: set[str] | None = None
        for depth, h in enumerate(block_hashes, start=1):
            owners = self._owners.get(h)
            if not owners:
                break
            if candidates is None:
                candidates = set(owners)
            else:
                candidates &= owners
            if not candidates:
                break
            for w in candidates:
                result[w] = depth
        return result

    def has_worker(self, worker_id: str) -> bool:
        return worker_id in self._worker_blocks

    def _discard(self, worker_id: str, block_hash: int) -> None:
        owners = self._owners.get(block_hash)
        if owners is None:
            return
        owners.discard(worker_id)
        if not owners:
            del self._owners[block_hash]
