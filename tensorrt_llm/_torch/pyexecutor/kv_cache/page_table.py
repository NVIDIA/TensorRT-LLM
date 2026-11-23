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

import enum
import types
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ...disaggregation.base.role_layout import ByHead as ByHead
from ...disaggregation.base.role_layout import Own as Own
from ...disaggregation.base.role_layout import RankTopology
from ...disaggregation.base.role_layout import Replicated as Replicated
from ...disaggregation.base.role_layout import RingSlots as RingSlots
from ...disaggregation.base.role_layout import RoleLayoutV2 as RoleLayoutV2
from ...disaggregation.base.role_layout import Sectioned as Sectioned

QUANT_MODES = frozenset({"fp8", "nvfp4", "fp8_ds_mla", "csa2_packed"})
"""The ways a spec says a manager's KV data is quantized (``PageTableSpec.quant_mode``)."""


def virtual_layers(
    attn_to_layer_id: Mapping[tuple[int, enum.Enum], int] | None,
) -> tuple[dict[int, tuple[int, int]], int] | None:
    """(internal id -> (model layer, attention type), type count) of a manager keeping several
    caches per model layer, from its (model layer, attention type) -> internal id map; None for an
    empty map. An internal id shared by several attention types of one model layer, as DeepSeek-V4
    keeps them, takes the first. A cache several model layers share enters the map only under the
    model layer that owns it, as CSA2's does, so its name does not depend on which layers a stage
    holds. Every member of the attention-type enum counts, so stages holding different attention
    types agree."""
    if not attn_to_layer_id:
        return None
    types_of: dict[int, tuple[int, int]] = {}
    for (model_layer, attn_type), layer in attn_to_layer_id.items():
        types_of.setdefault(int(layer), (int(model_layer), int(attn_type.value)))
    kinds = type(next(iter(attn_to_layer_id))[1])
    return types_of, max(int(member.value) for member in kinds) + 1


def layer_groups(manager: object) -> list[list[int]]:
    """Internal layer ids per runtime layer group, in layer-group order, each once in the order of
    the group's buffers; a layer whose buffers the reuse policies split is in each of its groups."""
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

    A manager returns it from ``page_table_spec``; ``build_page_table`` reads it with the
    manager's storage. The base manager's says K/V and their block scales split by KV head; a
    manager with other roles, or one of these laid out otherwise, returns its own, usually the
    base one with a few fields replaced (``dataclasses.replace``). Specs with equal fields
    compare equal; a spec holds mappings, so it is not hashable.

    Attributes:
        role_layouts: Role -> how its bytes spread over ranks.
        default_layout: The layout of a role ``role_layouts`` does not name, or None.
        untransferred_roles: Roles that hold nothing a peer needs: they stay in local copies of
            whole pages but never move.
        derived_roles: Roles derived from the transferred ones, which no transfer moves, native
            or through a lender: the receiver rebuilds them (``rebuild_disagg_imported_cache``)
            after a whole receive and up to the position a request resumes on a prefix it read.
        request_rebuilt_roles: Roles the manager, receiving, rebuilds for some requests rather
            than receiving them (``disagg_untransferred_buffers``), so a peer may hold none of
            them for a model layer it holds.
        layer_ids: Internal layer id -> (global layer id, model layer); None for those of the
            manager's model layers (``pp_layers``) or, keeping several caches per model layer,
            of its (model layer, attention type) map (``_layer_attn_to_layer_id``). Stated under
            mixed-precision KV, whose high layers take global layers ``2 * L + 1`` and every other
            layer twice the id it has without; the high layers hold the high type.
        quant_mode: How the manager's own KV data is quantized, as configured: None unquantized,
            else one of ``QUANT_MODES``: ``"fp8"``, ``"nvfp4"``, ``"fp8_ds_mla"`` (FP8 pages
            carrying their own scales) or ``"csa2_packed"`` (CSA2's fixed packed FP8/FP4 records,
            whatever its staging type). Under mixed-precision KV it is the high type's, as the
            manager's ``dtype`` is. It does not state per-layer element types, such as FP4 MLA's
            FP8 fallback layers or the low type of the layers kept mixed.
        mixed_precision: Reserved for the (low, high) element types of mixed-precision KV by
            lower-case ``DataType`` name (``"nvfp4"``; ``"half"``, ``"bf16"`` or ``"fp8"``), as
            ``("nvfp4", "bf16")``: a mixed layer keeps its history in rows of the low type, scales
            inline, and its first and last tokens in its high layer, of the high type. No manager
            fills it today, so it is None under mixed-precision KV too, and neither the page table
            nor a peer's check reads it.
        retention_margins: Whether the windows it keeps may hold margins of its own beyond the
            window its attention reads, such as draft tokens, a rewind or a lifecycle of their
            own, that the windows its layers declare they read (``_declared_read_windows``) do not
            leave out: they then do not say how far back it reads, and a transfer to it compares
            none of them (``layout_mapping.check_compatible``).

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


def layer_ids_of(manager: object, spec: PageTableSpec | None = None) -> dict[int, tuple[int, int]]:
    """Internal layer id -> (global layer id, model layer) of each layer of ``manager``'s layer
    groups, as its page table names them: the ``layer_ids`` of its spec (``spec``, else its
    ``page_table_spec()``) or, without them, its model layers (``_layer_ids``).

    Raises:
        ValueError: If its model layers name no global layer for some layer.
    """
    spec = manager.page_table_spec() if spec is None else spec
    if spec.layer_ids is not None:
        return dict(spec.layer_ids)
    internal = sorted({lid for group in layer_groups(manager) for lid in group})
    return _layer_ids(manager, internal)


def _layer_ids(manager: object, internal: list[int]) -> dict[int, tuple[int, int]]:
    """Internal layer id -> (global layer id, model layer), the same under any pipeline split:
    the model layer itself, or ``model_layer * num_types + type`` for a manager keeping several
    caches per model layer (``virtual_layers``)."""
    pp_layers = list(manager.pp_layers)
    virtual = virtual_layers(getattr(manager, "_layer_attn_to_layer_id", None))
    if virtual is None:
        unnamed = [lid for lid in internal if not 0 <= lid < len(pp_layers)]
        if unnamed:
            raise ValueError(
                f"internal layers {unnamed} are no local layers, and the manager names no model "
                "layer for them (_layer_attn_to_layer_id, PageTableSpec.layer_ids)"
            )
        # Internal layer i is the stage's i-th model layer, as layer_offsets numbers it.
        offsets = getattr(manager, "layer_offsets", None)
        if offsets is not None:
            renumbered = [lid for lid in internal if offsets.get(int(pp_layers[lid])) != lid]
            if renumbered:
                raise ValueError(
                    f"internal layers {renumbered} are not the local index layer_offsets gives "
                    "their model layers; a manager numbering its layers otherwise names them "
                    "(_layer_attn_to_layer_id, PageTableSpec.layer_ids)"
                )
        return {lid: (int(pp_layers[lid]),) * 2 for lid in internal}
    types_of, num_types = virtual
    unnamed = [lid for lid in internal if lid not in types_of]
    if unnamed:
        raise ValueError(
            f"internal layers {unnamed} have no (model layer, type) in _layer_attn_to_layer_id"
        )
    return {
        lid: (types_of[lid][0] * num_types + types_of[lid][1], types_of[lid][0]) for lid in internal
    }


def explicit_layer_specs(manager: object) -> dict[int, Any]:
    """Model layer -> the storage ``manager`` was given for it (``KVCacheLayerSpec``: its KV heads
    on this rank and the roles it stores): every layer's of a manager built from ``layer_specs``,
    else those of the drafter's layers a hybrid manager appends after its own
    (``_appended_draft_layer_specs``); none of a manager laying out its layers itself."""
    specs = getattr(manager, "layer_specs", None)
    if specs is None:
        specs = (getattr(manager, "_appended_draft_layer_specs", None) or {}).values()
    return {int(spec.global_layer_id): spec for spec in specs}
