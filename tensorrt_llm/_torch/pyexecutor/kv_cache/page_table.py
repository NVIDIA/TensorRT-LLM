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
``PageTableSpec``, such as how each role's bytes spread over ranks (``RoleLayoutV2``)."""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Mapping

# The role layouts a manager's spec names (disaggregation.base.role_layout), for managers to
# import from here.
from ...disaggregation.base.role_layout import ByHead as ByHead
from ...disaggregation.base.role_layout import Own as Own
from ...disaggregation.base.role_layout import Replicated as Replicated
from ...disaggregation.base.role_layout import RingSlots as RingSlots
from ...disaggregation.base.role_layout import RoleLayoutV2 as RoleLayoutV2
from ...disaggregation.base.role_layout import Sectioned as Sectioned


def layer_groups(manager: object) -> list[list[int]]:
    """Internal layer ids per cache group (``impl.buffer_grouping``), in cache-group order, each
    once in the order of the group's buffers; a layer whose buffers the reuse policies split is in
    each of its groups. Cache groups are not the deprecated layer groups (``impl.layer_grouping``)
    the ``get_disagg_*`` hooks index, which a split layer cannot be listed in."""
    return [
        list(dict.fromkeys(int(buffer.layer_id) for buffer in group))
        for group in manager.impl.buffer_grouping
    ]


# -- what a manager says of its pages: its PageTableSpec ------------------------------------------


def _frozen_roles(roles) -> frozenset[str]:
    return frozenset(str(role) for role in roles)


@dataclass(frozen=True, eq=False)
class PageTableSpec:
    """What a manager says of its pages that its runtime config and storage do not.

    A manager returns it from ``page_table_spec``. The base manager's says K/V and their block
    scales split by KV head; a manager with other roles, or one of these laid out otherwise,
    returns its own, usually the base one with a few fields replaced (``dataclasses.replace``).

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
        quant_mode: How the manager's own KV data is quantized: None unquantized, else
            ``"fp8"``, ``"nvfp4"`` or ``"fp8_ds_mla"``. Under mixed-precision KV, the layers it
            keeps mixed hold the types ``mixed_precision`` names instead.
        mixed_precision: Under mixed-precision KV, the (low, high) element types by lower-case
            ``DataType`` name (``"nvfp4"``; ``"half"``, ``"bf16"`` or ``"fp8"``), as
            ``("nvfp4", "bf16")``: a mixed layer keeps its history in rows of the low
            type, scales inline, and its first and last tokens in its high layer, of the high
            type; None without.

    Raises:
        ValueError: If mixed_precision does not name two types.
    """

    role_layouts: Mapping[str, RoleLayoutV2] = field(default_factory=dict)
    default_layout: RoleLayoutV2 | None = None
    untransferred_roles: frozenset[str] = frozenset()
    derived_roles: frozenset[str] = frozenset()
    request_rebuilt_roles: frozenset[str] = frozenset()
    layer_ids: Mapping[int, tuple[int, int]] | None = None
    quant_mode: str | None = None
    mixed_precision: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        layouts = {str(role): layout for role, layout in dict(self.role_layouts).items()}
        object.__setattr__(self, "role_layouts", types.MappingProxyType(layouts))
        for name in ("untransferred_roles", "derived_roles", "request_rebuilt_roles"):
            object.__setattr__(self, name, _frozen_roles(getattr(self, name)))
        if self.layer_ids is not None:
            ids = {int(lid): (int(g), int(m)) for lid, (g, m) in dict(self.layer_ids).items()}
            object.__setattr__(self, "layer_ids", types.MappingProxyType(ids))
        if self.mixed_precision is not None:
            names = tuple(str(name) for name in self.mixed_precision)
            if len(names) != 2:
                raise ValueError(f"mixed precision names a low and a high type, not {names}")
            object.__setattr__(self, "mixed_precision", names)

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
