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
"""What a ``KVCacheManagerV2`` says of its pages that its runtime config and storage do not: its
``PageTableSpec``, such as how each role's bytes spread over ranks (``RoleLayoutV2``); and what a
buffer (``BufferSite``) or a rank's place (``rank_topology``) says about it."""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ...disaggregation.base.role_layout import RankTopology

# The role layouts a manager's spec names (disaggregation.base.role_layout), for managers to
# import from here.
from ...disaggregation.base.role_layout import ByHead as ByHead
from ...disaggregation.base.role_layout import Own as Own
from ...disaggregation.base.role_layout import Replicated as Replicated
from ...disaggregation.base.role_layout import RingSlots as RingSlots
from ...disaggregation.base.role_layout import RoleLayoutV2 as RoleLayoutV2
from ...disaggregation.base.role_layout import Sectioned as Sectioned

__all__ = [
    "QUANT_MODES",
    "BufferSite",
    "ByHead",
    "Own",
    "PageTableSpec",
    "Replicated",
    "RingSlots",
    "RoleLayoutV2",
    "Sectioned",
    "kv_dtype_name",
    "layer_groups",
    "rank_topology",
    "state_split",
]

QUANT_MODES = frozenset({"fp8", "nvfp4", "fp8_ds_mla", "csa2_packed"})
"""The ways a spec says a manager's KV data is quantized (``PageTableSpec.quant_mode``)."""


def layer_groups(manager: object) -> list[list[int]]:
    """Internal layer ids per cache group (``impl.buffer_grouping``), in cache-group order, each
    once in the order of the group's buffers; a layer whose buffers the reuse policies split is in
    each of its groups. Cache groups are not the deprecated layer groups (``impl.layer_grouping``)
    the ``get_disagg_*`` hooks index, which a split layer cannot be listed in."""
    return [
        list(dict.fromkeys(int(buffer.layer_id) for buffer in group))
        for group in manager.impl.buffer_grouping
    ]


def rank_topology(mapping: object) -> RankTopology:
    """The rank's place as native's ``RankInfo`` reads it from the mapping: under attention data
    parallelism each tensor-parallel rank is a data-parallel rank of its own."""
    if mapping is None:
        return RankTopology()
    adp = bool(getattr(mapping, "enable_attention_dp", False))
    tp_size, tp_rank = int(getattr(mapping, "tp_size", 1)), int(getattr(mapping, "tp_rank", 0))
    return RankTopology(
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=int(getattr(mapping, "pp_size", 1)),
        pp_rank=int(getattr(mapping, "pp_rank", 0)),
        cp_size=int(getattr(mapping, "cp_size", 1)),
        cp_rank=int(getattr(mapping, "cp_rank", 0)),
        dp_size=tp_size if adp else int(getattr(mapping, "dp_size", 1)),
        dp_rank=tp_rank if adp else 0,
        attention_dp=adp,
    )


# -- what a manager says of its pages: its PageTableSpec ------------------------------------------


@dataclass(frozen=True)
class BufferSite:
    """One buffer as a manager's spec leaves it to its role layout (``RoleLayoutV2.resolve``).

    Attributes:
        layer: The manager's internal layer id.
        role: The buffer's role.
        size: Bytes of the buffer in one page's slot.
        tokens: Tokens of one page, or of one sub-page where the buffer has several per page.
        state: Whether the layer keeps recurrent state rather than attention KV.
        kv_heads: (KV heads on this rank, KV heads over all tensor ranks or None) of the layer.
        split: The parallel dimensions the layer's buffers split over: the tensor ranks for
            attention, those of the manager's recurrent state for state.
        ranks: Ranks in the grid of ``split``.
    """

    layer: int
    role: str
    size: int
    tokens: int
    state: bool
    kv_heads: tuple[int, int | None]
    split: tuple[str, ...]
    ranks: int


def _frozen_roles(roles: Iterable[str], name: str) -> frozenset[str]:
    """``roles`` as a frozen set of plain strings.

    Raises:
        TypeError: If ``roles`` is one role rather than a collection of them.
    """
    if isinstance(roles, str):
        raise TypeError(f"{name} is a collection of roles, not the role {roles!r}")
    return frozenset(str(role) for role in roles)


@dataclass(frozen=True)
class PageTableSpec:
    """What a manager says of its pages that its runtime config and storage do not.

    A manager returns it from ``page_table_spec``. The base manager's says K/V and their block
    scales split by KV head; a manager with other roles, or one of these laid out otherwise,
    returns its own, usually the base one with a few fields replaced (``dataclasses.replace``).
    Specs with equal fields compare equal; a spec holds mappings, so it is not hashable.

    Attributes:
        role_layouts: Role -> how its bytes spread over ranks.
        default_layout: The layout of a role ``role_layouts`` does not name, or None.
        untransferred_roles: Roles that hold nothing a peer needs: they stay in local copies of
            whole pages but never move.
        derived_roles: Roles derived from the transferred ones, which a transfer leaves for
            ``rebuild_disagg_imported_cache`` to rebuild after the import.
        request_rebuilt_roles: Roles the manager, receiving, rebuilds for some requests rather
            than receiving them, so a peer may hold none of them for a model layer it holds.
        layer_ids: Internal layer id -> (global layer id, model layer) of every layer the
            manager's storage holds, as disaggregated transfer names them: a layer's model layer
            (``pp_layers``) or, keeping several caches per model layer, model layer x attention
            types + attention type (``_layer_attn_to_layer_id``); under mixed-precision KV, high
            layers ``2 * L + 1``, holding the high type (``mixed_precision``), and every other
            layer twice its id; with the FP4 MLA cache layout, a model layer's cache ``2 * m``,
            its high-precision tail ``2 * m + 1`` and a recurrent layer ``2 * m``. Every
            manager's spec states it (``_disagg_layer_names``); None only in a spec made without
            a manager.
        quant_mode: How the manager's own KV data is quantized, as configured: None unquantized,
            else one of ``QUANT_MODES``: ``"fp8"``, ``"nvfp4"``, ``"fp8_ds_mla"`` (FP8 pages
            carrying their own scales) or ``"csa2_packed"`` (CSA2's fixed packed FP8/FP4 records,
            whatever its staging type). Under mixed-precision KV, the layers it keeps mixed hold
            the types ``mixed_precision`` names instead. It does not state per-layer element
            types, such as FP4 MLA's FP8 fallback layers.
        mixed_precision: Under mixed-precision KV, the (low, high) element types by lower-case
            ``DataType`` name (``"nvfp4"``; ``"half"``, ``"bf16"`` or ``"fp8"``), as
            ``("nvfp4", "bf16")``: a mixed layer keeps its history in rows of the low
            type, scales inline, and its first and last tokens in its high layer, of the high
            type. Under the cold-page first/last-N window both are the model's KV type, as
            ``("bf16", "bf16")``: the high layer holds a copy of its layer's page, and only pages
            leaving the GPU are quantized. None without.
        retention_margins: Whether the windows it keeps may hold margins of its own beyond the
            window its attention reads, such as draft tokens, a rewind or a lifecycle of their
            own: they then do not say how far back it reads.

    Raises:
        TypeError: If a role set is one role, or a layout is not a ``RoleLayoutV2``.
        ValueError: If a role is both untransferred and derived, two layers share a global layer
            id, ``quant_mode`` is not one of ``QUANT_MODES`` or ``mixed_precision`` does not
            name two types.
    """

    role_layouts: Mapping[str, RoleLayoutV2] = field(default_factory=dict)
    default_layout: RoleLayoutV2 | None = None
    untransferred_roles: frozenset[str] = frozenset()
    derived_roles: frozenset[str] = frozenset()
    request_rebuilt_roles: frozenset[str] = frozenset()
    layer_ids: Mapping[int, tuple[int, int]] | None = None
    quant_mode: str | None = None
    mixed_precision: tuple[str, str] | None = None
    retention_margins: bool = False

    def __post_init__(self) -> None:
        layouts = {str(role): layout for role, layout in dict(self.role_layouts).items()}
        for layout in (*layouts.values(), self.default_layout):
            if layout is not None and not isinstance(layout, RoleLayoutV2):
                raise TypeError(f"a role layout is a RoleLayoutV2, not {layout!r}")
        object.__setattr__(self, "role_layouts", types.MappingProxyType(layouts))
        for name in ("untransferred_roles", "derived_roles", "request_rebuilt_roles"):
            object.__setattr__(self, name, _frozen_roles(getattr(self, name), name))
        if both := self.untransferred_roles & self.derived_roles:
            raise ValueError(f"roles both untransferred and derived: {sorted(both)}")
        if self.layer_ids is not None:
            ids = {int(lid): (int(g), int(m)) for lid, (g, m) in dict(self.layer_ids).items()}
            global_layers = [global_layer for global_layer, _ in ids.values()]
            if len(set(global_layers)) != len(global_layers):
                raise ValueError(f"layers share a global layer id: {sorted(global_layers)}")
            object.__setattr__(self, "layer_ids", types.MappingProxyType(ids))
        if self.quant_mode is not None and self.quant_mode not in QUANT_MODES:
            raise ValueError(f"quant_mode {self.quant_mode!r} is not one of {sorted(QUANT_MODES)}")
        if self.mixed_precision is not None:
            names = tuple(str(name) for name in self.mixed_precision)
            if len(names) != 2:
                raise ValueError(f"mixed precision names a low and a high type, not {names}")
            object.__setattr__(self, "mixed_precision", names)
        object.__setattr__(self, "retention_margins", bool(self.retention_margins))

    def layout_of(self, role: str) -> RoleLayoutV2 | None:
        """The layout of ``role``: this rank's own share for a role that never moves (untransferred
        or derived), whatever the spec lays out otherwise, else its own, else the default; None for
        a role that moves and that the spec does not lay out. A transfer leaves the roles that
        never move by their membership in those sets, not by their layout."""
        if role in self.untransferred_roles | self.derived_roles:
            return Own()
        return self.role_layouts.get(role, self.default_layout)


def kv_dtype_name(manager: object) -> str | None:
    """The element type of a manager's KV data by name, such as ``"BF16"``; None if it has
    none."""
    dtype = getattr(manager, "dtype", None)
    return None if dtype is None else str(getattr(dtype, "name", dtype))


def state_split(manager: object) -> tuple[str, ...]:
    """The parallel dimensions a manager's recurrent state splits over, as its pools are sized
    (``config_utils.mamba_effective_tp_size``): tensor times context ranks under Helix, and under
    Helix with attention data parallelism on one tensor rank its context ranks; otherwise the
    tensor ranks of an attention data-parallel group (``grid_place``), which TRT-LLM's mappings
    make one rank alone under attention data parallelism, so the state is then whole on every
    rank."""
    mapping = manager.mapping
    adp = bool(getattr(mapping, "enable_attention_dp", False))
    if getattr(manager, "_has_cp_helix", False):
        if not adp:
            return ("tp", "cp")
        if int(getattr(mapping, "tp_size", 1)) == 1:
            return ("cp",)
    return ("tp",)
