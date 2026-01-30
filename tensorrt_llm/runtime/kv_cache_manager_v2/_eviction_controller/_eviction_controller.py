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

from typing import Callable, Protocol, cast

from llist import sllist, sllistnode

from .._common import NDEBUG, CacheLevel, PageStatus, Priority
from .._exceptions import OutOfPagesError
from .._life_cycle_registry import LifeCycleId
from .._storage._core import PoolGroupIndex
from .._utils import (
    TypedIndexList,
    assert_critical,
    make_typed,
    noexcept,
    typed_enumerate,
    typed_len,
    unwrap_optional,
)


# @runtime_checkable
class EvictablePage(Protocol):
    @property
    def cache_level(self) -> CacheLevel: ...

    @property
    def priority(self) -> Priority: ...

    @property
    def life_cycle(self) -> LifeCycleId: ...

    @property
    def status(self) -> PageStatus: ...

    def is_committed(self) -> bool: ...

    node_ref: "NodeRef | None"


# @runtime_checkable
class NodeRef(Protocol):
    @property
    def value(self) -> EvictablePage: ...


# @runtime_checkable
class EvictionPolicy(Protocol):
    def push(self, page: EvictablePage, evict_first: bool = False) -> NodeRef: ...

    def pop(self) -> EvictablePage: ...

    # Remove a node so we no longer consider it for eviction. Like pop() but allow removing a node
    # that is not the first.
    def remove(self, node: NodeRef) -> EvictablePage: ...

    def __len__(self) -> int: ...


class LRUEvictionPolicy:
    __slots__ = ("_queue",)
    _queue: sllist

    def __init__(self) -> None:
        self._queue = sllist()

    def push(self, page: EvictablePage, evict_first: bool = False) -> sllistnode:
        assert page.node_ref is None
        return self._queue.appendleft(page) if evict_first else self._queue.append(page)

    def pop(self) -> EvictablePage:
        victim = self._queue.first
        assert victim is not None
        page = victim.value
        self.remove(victim)
        return page

    def remove(self, node: sllistnode) -> EvictablePage:
        # assert isinstance(node, NodeRef) # mypyc does not support runtime_checkable
        assert node == node.value.node_ref
        return self._queue.remove(node)

    def __len__(self) -> int:
        return len(self._queue)


# helper class to help add support for priority-based eviction
class PrioritizedEvictionPolicy:
    __slots__ = (
        "_policy_creator",
        "_policies",
    )
    _policy_creator: Callable[[Priority], EvictionPolicy]
    _policies: dict[Priority, EvictionPolicy]

    def __init__(self, policy_creator: Callable[[Priority], EvictionPolicy]) -> None:
        self._policy_creator = policy_creator
        self._policies = {}

    def __len__(self) -> int:
        return sum(len(policy) for policy in self._policies.values())

    def get_policy(self, priority: Priority) -> EvictionPolicy:
        if priority not in self._policies:
            self._policies[priority] = self._policy_creator(priority)
            self._policies = dict(sorted(self._policies.items()))
        return self._policies[priority]

    def _front_policy(self) -> EvictionPolicy:
        return next(iter(self._policies.values()))

    def push(self, page: EvictablePage, evict_first: bool = False) -> NodeRef:
        return self.get_policy(page.priority).push(page, evict_first)

    def pop(self) -> EvictablePage:
        return self._front_policy().pop()

    def remove(self, node: NodeRef) -> EvictablePage:
        page = node.value
        policy = self._policies[page.priority]
        policy.remove(node)
        if not policy:
            self._policies.pop(page.priority)
        return page


class PrioritizedLRUEvictionPolicy(PrioritizedEvictionPolicy):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(lambda priority: LRUEvictionPolicy())


class PerLevelEvictionController:  # for one cache level
    __slots__ = ("_life_cycle_grouping", "_policies", "_cache_level")
    _life_cycle_grouping: TypedIndexList[LifeCycleId, PoolGroupIndex]
    _policies: TypedIndexList[PoolGroupIndex, EvictionPolicy]
    _cache_level: CacheLevel

    def __init__(
        self,
        life_cycle_grouping: TypedIndexList[LifeCycleId, PoolGroupIndex],
        cache_level: CacheLevel,
    ):
        self._cache_level = cache_level
        self._life_cycle_grouping = life_cycle_grouping
        num_pool_groups = max(life_cycle_grouping) + 1
        assert num_pool_groups == len(set(life_cycle_grouping))
        self._policies = cast(
            TypedIndexList, [PrioritizedLRUEvictionPolicy() for _ in range(num_pool_groups)]
        )

    def __del__(self) -> None:
        if not NDEBUG:
            assert_critical(
                all(len(p) == 0 for p in self._policies), "Eviction controller is not empty"
            )

    def _get_policy(self, life_cycle: LifeCycleId) -> EvictionPolicy:
        pg_idx = self._life_cycle_grouping[life_cycle]
        return self._policies[pg_idx]

    def schedule_for_eviction(self, page: EvictablePage, evict_first: bool = False):
        assert page.node_ref is None and page.cache_level == self._cache_level
        page.node_ref = self._get_policy(page.life_cycle).push(page, evict_first)
        assert unwrap_optional(page.node_ref).value is page

    # If evicting a node makes some other nodes useless, those nodes will be returned as well.
    # One example: for SWA, if the number of blocks just makes up one window size, then evicting any of
    # them makes the remaining blocks useless.
    # Raise if no enough pages to evict. In this case, pages are returned to the eviction queue.
    def evict(
        self, min_num_pages: TypedIndexList[PoolGroupIndex, int]
    ) -> TypedIndexList[PoolGroupIndex, list[EvictablePage]]:
        assert NDEBUG or len(min_num_pages) == self.num_pool_groups
        ret = make_typed(lambda: list[EvictablePage](), self.num_pool_groups)
        try:
            for pg_idx, count in typed_enumerate(min_num_pages):
                policy = self._policies[pg_idx]
                if (len(policy) + len(ret[pg_idx])) < count:
                    raise OutOfPagesError(f"Not enough pages to evict in group {pg_idx}")
                while len(ret[pg_idx]) < count:
                    page = policy.pop()
                    page.node_ref = None
                    ret[pg_idx].append(page)
                    for a, b in zip(ret, self._evict_dependencies(page)):
                        a.extend(b)
        except Exception:
            for p in reversed(sum(ret, [])):
                self.schedule_for_eviction(p, evict_first=True)
            raise
        assert all(p.cache_level == self._cache_level for p in sum(ret, [])), (
            "Corrupted eviction controller"
        )
        return ret

    def remove(self, node: NodeRef) -> None:
        page = node.value
        assert page.node_ref == node
        self._get_policy(page.life_cycle).remove(node)
        page.node_ref = None

    # @TODO: implement this
    @noexcept
    def _evict_dependencies(
        self, page: EvictablePage
    ) -> TypedIndexList[PoolGroupIndex, list[EvictablePage]]:
        return make_typed(lambda: list[EvictablePage](), self.num_pool_groups)

    def num_evictable_pages(self, pg_idx: PoolGroupIndex) -> int:
        return len(self._policies[pg_idx])

    @property
    def num_pool_groups(self) -> PoolGroupIndex:
        return typed_len(self._policies)
