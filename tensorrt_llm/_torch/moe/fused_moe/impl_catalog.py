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
"""The user-facing catalog of the MoE implementations a tactic can pin.

The MoE performance guide embeds what :func:`moe_impl_catalog_markdown`
renders between :data:`CATALOG_BEGIN` and :data:`CATALOG_END`, and
``scripts/generate_moe_impl_catalog.py`` rewrites that block, so the
reference users read follows the registry rather than a hand-kept list.
"""

from dataclasses import fields

from .impl_identity import MOE_IMPL_REGISTRY, MoEImplId
from .moe_resolution import BACKEND_FAMILY, IMPL_PRIORITY, MoEImplClass, backend_family_of

CATALOG_BEGIN = "<!-- BEGIN GENERATED MOE IMPL CATALOG: scripts/generate_moe_impl_catalog.py -->"
CATALOG_END = "<!-- END GENERATED MOE IMPL CATALOG -->"


def _ranked_impls() -> list[tuple[int, MoEImplClass, MoEImplId]]:
    """Every registered implementation with its 1-based rank in ``IMPL_PRIORITY``."""
    registered = {
        MOE_IMPL_REGISTRY.lookup(identity): identity for identity in MOE_IMPL_REGISTRY.identities()
    }
    ranked = [impl_cls for impl_cls in IMPL_PRIORITY if impl_cls in registered]
    unranked = sorted(impl_cls.__name__ for impl_cls in registered.keys() - set(ranked))
    if unranked:
        raise RuntimeError(
            f"MoE impls are registered but absent from IMPL_PRIORITY, so the catalog "
            f"cannot rank them: {unranked}"
        )
    return [(rank, impl_cls, registered[impl_cls]) for rank, impl_cls in enumerate(ranked, 1)]


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def moe_impl_catalog_markdown() -> str:
    """The catalog: one row per implementation id, then the segment values.

    Rows are grouped by backend family, in ``BACKEND_FAMILY`` order, so the
    rows of one ``moe_config.backend`` value sit together. The shortest tactic
    is :meth:`MoEImplRegistry.shortest_query`, the fewest segments naming that
    implementation alone. Priority is the global rank resolution tries the
    matches of a tactic in.
    """
    family_order = {name: index for index, name in enumerate(BACKEND_FAMILY)}

    def family_then_rank(entry: tuple[int, MoEImplClass, MoEImplId]) -> tuple[int, int]:
        rank, impl_cls, _ = entry
        family = backend_family_of(impl_cls)
        return (family_order.get(family, len(family_order)), rank)

    lines = [
        "| Implementation id | Shortest tactic | `backend` family | Priority | Description |",
        "|---|---|---|---|---|",
    ]
    for rank, impl_cls, identity in sorted(_ranked_impls(), key=family_then_rank):
        family = backend_family_of(impl_cls)
        family_cell = f"`{family}`" if family is not None else "none (impl_id only)"
        shortest = MOE_IMPL_REGISTRY.shortest_query(identity).compact()
        lines.append(
            f"| `{identity}` | `{shortest}` | {family_cell} | {rank} "
            f"| {_cell(impl_cls.descriptor.doc)} |"
        )

    lines += ["", "Segment values a tactic can use, by field:", ""]
    tokens = MOE_IMPL_REGISTRY.known_tokens()
    for field in fields(MoEImplId):
        values = sorted(token for token, owner in tokens.items() if owner == field.name)
        lines.append(f"- `{field.name}`: " + ", ".join(f"`{value}`" for value in values))
    return "\n".join(lines) + "\n"


def splice_moe_impl_catalog(document: str) -> str:
    """``document`` with the block between the catalog markers regenerated.

    Raises:
        ValueError: ``document`` does not hold each marker exactly once, in order.
    """
    if document.count(CATALOG_BEGIN) != 1 or document.count(CATALOG_END) != 1:
        raise ValueError(f"expected exactly one {CATALOG_BEGIN!r} and one {CATALOG_END!r} marker")
    head, rest = document.split(CATALOG_BEGIN)
    if CATALOG_END not in rest:
        raise ValueError(f"{CATALOG_END!r} must follow {CATALOG_BEGIN!r}")
    _, tail = rest.split(CATALOG_END)
    return f"{head}{CATALOG_BEGIN}\n\n{moe_impl_catalog_markdown()}\n{CATALOG_END}{tail}"
