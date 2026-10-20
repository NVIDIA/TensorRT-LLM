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
"""The base of every built-in kind: a ``KVCacheManagerV2`` described life cycle by life cycle.

Each life cycle becomes one unit space (its index is the life-cycle id), paged or state as
:meth:`StandardSpec.life_cycle_unit` decides, and each ``(layer, role)`` buffer in it one region
whose layout :meth:`StandardSpec.role_buffers` declares. The defaults describe standard paged
attention: K/V sharded by KV head, head-major, and the index-key role replicated. Roles the manager
declares storage-only (``get_disagg_ignored_roles``) get no region. Selection and memory rules follow
the space's unit (``_paged`` / ``_state``). A variant subclasses this and overrides what its cache
does differently; it is not registered itself.
"""

from __future__ import annotations

from typing import FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind

from ..page_table import BufferDecl, PageTable, Region, UnitSpace
from ..spec import (
    CacheKindSpec,
    ChunkContext,
    LocalUnitsContext,
    MatchedPair,
    PairContext,
    RegistrationRegion,
)
from . import _ownership, _paged, _state, _v2
from ._v2 import LifeCycle


class StandardSpec(CacheKindSpec):
    # ---- description ----

    def describe(self, manager) -> PageTable:
        if not (hasattr(manager, "impl") and hasattr(manager.impl, "layer_grouping")):
            raise ValueError(
                f"{type(manager).__name__} is not a KVCacheManagerV2: the Python KV transceiver "
                "transfers V2 caches only (use kv_cache_config.use_kv_cache_manager_v2=True, or "
                "transceiver_runtime='CPP' for a V1 manager)"
            )
        ignored = _v2.ignored_roles(manager)
        pools, cycles = _v2.life_cycles(manager)
        spaces = [self.describe_life_cycle(manager, cycle, ignored) for cycle in cycles]
        return PageTable(pools=pools, spaces=spaces)

    def describe_life_cycle(
        self, manager, cycle: LifeCycle, ignored: FrozenSet[str] = frozenset()
    ) -> UnitSpace:
        unit = self.life_cycle_unit(manager, cycle)
        tokens_per_block = int(manager.impl.init_config.tokens_per_block)
        heads = _paged.kv_heads_per_rank(manager)
        global_ids = dict(zip(cycle.internal_layer_ids, self.global_layer_ids(manager, cycle)))
        regions = []
        for buf in cycle.buffers:
            # Ignored roles still occupy slot offsets but carry no state to transfer.
            if buf.role in ignored:
                continue
            if buf.local_layer not in global_ids:
                raise ValueError(
                    f"life cycle {cycle.index} holds a buffer of layer {buf.local_layer}, which is "
                    f"not one of its layers {cycle.internal_layer_ids}"
                )
            buffers = self.role_buffers(
                manager,
                unit,
                buf.role,
                buf.size,
                heads=heads,
                tokens_per_block=tokens_per_block,
                context=f"{self.name} life cycle {cycle.index} role {buf.role!r}",
            )
            regions.append(
                Region(
                    pool=buf.pool,
                    layer=int(global_ids[buf.local_layer]),
                    role=buf.role,
                    offset=buf.offset,
                    buffers=buffers,
                    local_layer=buf.local_layer,
                )
            )
        regions.sort(key=lambda r: (r.pool, r.offset))
        if unit == CacheKind.PAGED:
            params = _paged.params(tokens_per_block, self.window_size(manager, cycle))
        else:
            params = {}
        return UnitSpace(kind=self.name, unit=unit, regions=regions, params=params)

    def role_buffers(
        self,
        manager,
        unit: CacheKind,
        role: str,
        nbytes: int,
        *,
        heads: int,
        tokens_per_block: int,
        context: str,
    ) -> Tuple[BufferDecl, ...]:
        """How one ``(layer, role)`` buffer of ``nbytes`` is laid out and split across ranks.

        ``heads`` is the rank's KV head count. By default a role in :meth:`replicated_roles` moves
        whole, other paged roles are K/V sharded by head in :meth:`kv_layout`, and state roles are
        laid out by :meth:`state_buffers`.
        """
        if role in self.replicated_roles(manager):
            if unit == CacheKind.PAGED:
                return (_paged.replicated(nbytes),)
            return _state.whole(nbytes)
        if unit == CacheKind.PAGED:
            return (_paged.kv(nbytes, heads, tokens_per_block, self.kv_layout(manager), context),)
        return self.state_buffers(manager, role, nbytes, context)

    def replicated_roles(self, manager) -> FrozenSet[str]:
        """Roles every rank holds identically. Index-key caches are computed replicated across TP
        (a single index head), so their bytes are the same on every rank."""
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role

        return frozenset({str(Role.INDEX_KEY)})

    def kv_layout(self, manager) -> str:
        """How the attention backend paired with the manager stores K/V in a block."""
        return _paged.HEAD_MAJOR

    def state_buffers(self, manager, role: str, nbytes: int, context: str):
        """The layout of a state role not in :meth:`replicated_roles`; held whole by default."""
        return _state.whole(nbytes)

    def life_cycle_unit(self, manager, cycle: LifeCycle) -> CacheKind:
        """Whether a life cycle's ids are blocks (PAGED) or one slot per request (STATE)."""
        return CacheKind.PAGED

    def global_layer_ids(self, manager, cycle: LifeCycle) -> List[int]:
        """Collision-free ids of a life cycle's layers, equal on every PP split.

        They are not model layer indices; they only guarantee that different internal layers get
        different ids and that one internal layer gets the same id whatever the PP split.
        """
        return _v2.default_global_layer_ids(manager, cycle)

    def window_size(self, manager, cycle: LifeCycle) -> Optional[int]:
        return _v2.window_size(manager, cycle.internal_layer_ids[0])

    def requires_physical_ownership(self, manager, mapping, transceiver_config) -> bool:
        return _ownership.requires_physical_ownership(manager, mapping, transceiver_config)

    def state_slot(self, manager, request) -> Optional[int]:
        """The request's slot in a state space, or None when this rank holds no state."""
        raise NotImplementedError(f"{self.name} describes no state units")

    # ---- selection ----

    def local_units(self, ctx: LocalUnitsContext) -> np.ndarray:
        if ctx.space.unit == CacheKind.PAGED:
            return _paged.local_units(ctx)
        return _state.units(self.state_slot(ctx.cache_manager, ctx.request))

    def chunk_units(self, space: UnitSpace, units: np.ndarray, ctx: ChunkContext) -> np.ndarray:
        if space.unit == CacheKind.PAGED:
            return _paged.chunk_units(space, units, ctx)
        # Whole-request units ride the last chunk.
        return units if ctx.is_last else np.full_like(units, -1)

    def pair_units(self, src_units, dst_units, ctx: PairContext):
        if ctx.space.unit == CacheKind.PAGED:
            return _paged.pair_units(src_units, dst_units, ctx)
        return _state.pair_units(src_units, dst_units, ctx)

    def transfer_bytes(self, space: UnitSpace, region_bytes: int, units: np.ndarray) -> int:
        if space.unit == CacheKind.PAGED:
            return _paged.transfer_bytes(region_bytes, units)
        return _state.transfer_bytes(region_bytes, units)

    def sends_incrementally(self, space: UnitSpace) -> bool:
        return space.unit == CacheKind.PAGED

    # ---- compatibility ----

    def check_rank_compatible(self, self_ri, peer_ri) -> bool:
        return _paged.check_rank_compatible(self.name, self_ri, peer_ri)

    def validate_peer(self, self_ri, peer_ri, pairs: Sequence[MatchedPair]) -> None:
        _state.validate_peer(self.name, self_ri, peer_ri, pairs)

    def on_peer_registered(self, self_ri, peer_ri, groups) -> None:
        _paged.warn_token_major_fragments(self_ri, peer_ri, groups)

    # ---- memory ----

    def registration_regions(
        self, page_table: PageTable, space_index: int
    ) -> List[RegistrationRegion]:
        if page_table.spaces[space_index].unit == CacheKind.PAGED:
            return _paged.registration_regions(page_table, space_index)
        return _state.registration_regions(page_table, space_index)

    def bounce_unit_bytes(self, page_table: PageTable, space_index: int) -> Optional[int]:
        if page_table.spaces[space_index].unit == CacheKind.PAGED:
            return _paged.bounce_unit_bytes(page_table, space_index)
        return None

    def receiver_extra_bytes(self, sender_page_table, receiver_page_table, space_index, dst_units):
        if receiver_page_table.spaces[space_index].unit == CacheKind.PAGED:
            return 0
        return _state.receiver_extra_bytes(
            sender_page_table, receiver_page_table, space_index, dst_units
        )
