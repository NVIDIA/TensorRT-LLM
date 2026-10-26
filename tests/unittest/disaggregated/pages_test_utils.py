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
"""Hand-built ``RankPages`` for tests: pools by flat index, layer groups of buffers in them.

``pages([pool(...), ...], [group(..., [buf(...), ...]), ...])`` builds the page layout a cache
manager would report. Pools a layer group's buffers name are gathered into one pool group, in the
order given; layer groups naming the same pool share its pool group, as storage shares a pool
group between life cycles of equal slot sizes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from tensorrt_llm._torch.disaggregation.page_mapping import RankPages
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_builder import DevicePool
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    RankTopology,
    Segment,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.role_layout import (  # noqa: F401
    head_major,
    replicated,
    sectioned,
    token_major,
)

__all__ = [
    "Segment",
    "buf",
    "group",
    "head_major",
    "pages",
    "pool",
    "replicated",
    "sectioned",
    "token_major",
]


@dataclass(frozen=True)
class pool:  # noqa: N801  (reads like the other builders)
    base: int
    slot_bytes: int
    num_slots: int = 16


@dataclass(frozen=True)
class buf:  # noqa: N801
    """One buffer: global ``layer`` and ``role`` at ``offset`` in a slot of flat pool ``pool``."""

    pool: int
    layer: int
    role: str
    offset: int
    segments: Tuple[Segment, ...]
    transferred: bool = True

    @property
    def size(self) -> int:
        return sum(s.nbytes for s in self.segments)


@dataclass(frozen=True)
class group:  # noqa: N801
    """One layer group: ``"attention"`` (paged) or ``"state"`` (one slot per request).

    ``grid``, ``unit`` and ``blocks_by_cp`` default from ``kind`` (``page_layout.KIND_DEFAULTS``).
    """

    kind: str
    buffers: Sequence[buf] = field(default_factory=tuple)
    window: Optional[int] = None
    grid: Optional[Tuple[str, ...]] = None
    unit: Optional[str] = None
    blocks_by_cp: Optional[bool] = None


def _pool_groups(groups: Sequence[group]) -> Dict[int, Tuple[int, int]]:
    """Flat pool -> (pool group, pool index): pools named by one layer group join one pool group."""
    parent: Dict[int, int] = {}

    def find(p: int) -> int:
        while parent.setdefault(p, p) != p:
            p = parent[p]
        return p

    for g in groups:
        named = sorted({b.pool for b in g.buffers})
        for p in named:
            parent[find(p)] = find(named[0])
    members: Dict[int, List[int]] = {}
    for p in sorted(parent):
        members.setdefault(find(p), []).append(p)
    out = {}
    for pool_group, root in enumerate(sorted(members)):
        for index, p in enumerate(members[root]):
            out[p] = (pool_group, index)
    return out


def pages(
    pools: Sequence[pool],
    groups: Sequence[group],
    *,
    tokens_per_page: int = 4,
    topology: RankTopology = RankTopology(),
    total_kv_heads_by_layer: Optional[Dict[int, int]] = None,
) -> RankPages:
    where = _pool_groups(groups)
    for p in range(len(pools)):
        where.setdefault(p, (len({g for g, _ in where.values()}), 0))
    layer_groups = []
    for g in groups:
        pool_groups = {where[b.pool][0] for b in g.buffers}
        if len(pool_groups) > 1:
            raise ValueError(f"layer group spans pool groups {pool_groups}")
        layers = tuple(sorted({b.layer for b in g.buffers}))
        layer_groups.append(
            LayerGroupDesc(
                kind=g.kind,
                kept_window=g.window,
                num_sink_blocks=0,
                pool_group=pool_groups.pop() if pool_groups else 0,
                layers=layers,
                grid=g.grid,
                unit=g.unit,
                blocks_by_cp=g.blocks_by_cp,
            )
        )
    buffers = tuple(
        BufferDesc(
            layer=b.layer,
            role=b.role,
            pool_group=where[b.pool][0],
            pool_index=where[b.pool][1],
            slot_offset=b.offset,
            size=b.size,
            segments=b.segments,
            transferred=b.transferred,
        )
        for g in groups
        for b in g.buffers
    )
    slot_bytes: Dict[int, List[int]] = {}
    device = []
    for p, spec in enumerate(pools):
        pool_group, index = where[p]
        slot_bytes.setdefault(pool_group, []).append(spec.slot_bytes)
        device.append(DevicePool(pool_group, index, spec.base, spec.slot_bytes, spec.num_slots))
    layout = PageLayout(
        tokens_per_page=tokens_per_page,
        pool_slot_bytes={g: tuple(w) for g, w in slot_bytes.items()},
        layer_groups=tuple(layer_groups),
        buffers=buffers,
        topology=topology,
        total_kv_heads_by_layer=total_kv_heads_by_layer or {},
    )
    return RankPages(layout, tuple(device))
