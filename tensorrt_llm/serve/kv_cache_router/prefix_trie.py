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
"""Block-hash prefix index for KV-cache-aware routing.

A KV-cache block hash folds in its parent chain (the worker computes it with
``BlockKeyHasher.hash(block_key, parent_hash)``), so every block-hash *value* is
globally unique to one position in one prefix path. A request's ordered
block-hash list is itself the prefix path, so longest-common-prefix against a set
of held blocks is just "walk the list until the first hash the owner doesn't
hold" -- no explicit tree needed.

The centralized router keeps one :class:`PrefixBlockSet` per single logical owner
(a per-rank set, and a per-instance combined set). This mirrors the orchestrator
``KvCacheAwareServerState``, which keeps one ``set[block_hash]`` per server and
matches with the same walk. All operations are O(L) in the number of query /
changed blocks.
"""

from typing import Iterable, List

__all__ = ["PrefixBlockSet"]


class PrefixBlockSet:
    """Single-owner block-hash index -- a flat ``set`` of held block hashes.

    This is the exact structure the orchestrator ``KvCacheAwareServerState`` uses
    (a ``set[block_hash]`` per server, walked until the first miss). It is the
    right index whenever there is only ONE logical owner -- e.g. the centralized
    router's per-instance ``combined_trie`` (owner = instance) and each rank's
    trie (owner = that rank). Those are only ever queried for a single owner's
    prefix depth (:meth:`match_one`), so a ``hash -> {owner}`` reverse map with
    per-depth set intersections would be pure overhead here: a one-element
    ``set()`` allocated per block hash and maintained on every add/remove, for
    data that is never read.

    The ``owner_id`` argument on :meth:`add` / :meth:`remove` / :meth:`match_one`
    is accepted and ignored so this is a drop-in for the single-owner call sites.
    """

    __slots__ = ("_blocks",)

    def __init__(self) -> None:
        self._blocks: set[int] = set()

    def add(self, owner_id: str, block_hashes: Iterable[int]) -> None:
        self._blocks.update(block_hashes)

    def remove(self, owner_id: str, block_hashes: Iterable[int]) -> None:
        self._blocks.difference_update(block_hashes)

    def remove_worker(self, owner_id: str) -> None:
        self._blocks.clear()

    def match_one(self, owner_id: str, block_hashes: List[int]) -> int:
        """Consecutive prefix-block count held by the (single) owner.

        Identical to ``KvCacheAwareServerState.matched_tokens``: walk the query
        path, counting blocks present in the set, and stop at the first miss.
        """
        blocks = self._blocks
        depth = 0
        for h in block_hashes:
            if h not in blocks:
                break
            depth += 1
        return depth

    def has_worker(self, owner_id: str) -> bool:
        return bool(self._blocks)
