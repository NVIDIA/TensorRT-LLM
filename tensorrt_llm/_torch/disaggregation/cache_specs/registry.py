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
"""The registered cache specs, and the only place a spec is identified.

A manager is claimed by the spec that names the class nearest to it in its MRO; a unit space names
its spec (``UnitSpace.spec``), which maps to the registered spec here. A name no spec registers is
an error wherever it is met -- validating a peer, matching regions, building a transfer -- never a
skip.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Type, TypeVar

from .page_table import PageTable, UnitSpace
from .spec import CacheSpec

_SPECS: Dict[str, CacheSpec] = {}

DEFAULT_SPEC = "attention"
"""Describes a manager no spec claims by class (paged attention on KVCacheManagerV2)."""

_SpecT = TypeVar("_SpecT", bound=Type[CacheSpec])


def register(spec_cls: _SpecT) -> _SpecT:
    """Class decorator: register one instance of a :class:`CacheSpec` subclass."""
    name = spec_cls.name
    existing = _SPECS.get(name)
    if existing is not None and type(existing) is not spec_cls:
        raise ValueError(f"cache spec name {name!r} is already registered")
    for other in _SPECS.values():
        shared = set(other.managers) & set(spec_cls.managers)
        if type(other) is not spec_cls and shared:
            raise ValueError(f"{sorted(shared)} already claimed by cache spec {other.name!r}")
    _SPECS[name] = spec_cls()
    return spec_cls


def unregister(spec_cls: Type[CacheSpec]) -> None:
    """Remove a spec registered by :func:`register` (test-only specs)."""
    if type(_SPECS.get(spec_cls.name)) is spec_cls:
        del _SPECS[spec_cls.name]


def registered_specs() -> List[CacheSpec]:
    return list(_SPECS.values())


def spec_named(name: str) -> CacheSpec:
    try:
        return _SPECS[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown cache spec {name!r} (known: {sorted(_SPECS)}); the peer runs a cache spec "
            "this build cannot transfer"
        ) from None


def spec_for(space: UnitSpace) -> CacheSpec:
    """The spec of one unit space."""
    return spec_named(space.spec)


def _resolve(path: str) -> Optional[type]:
    """The class at a dotted path, if its module is loaded (an unloaded class has no instances)."""
    module_name, _, qualname = path.rpartition(".")
    obj = sys.modules.get(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part, None) if obj is not None else None
    return obj if isinstance(obj, type) else None


def claim(manager) -> CacheSpec:
    """The spec that describes ``manager``: the one naming the class nearest in its MRO."""
    # ``__class__`` rather than ``type()`` so a spec'd mock is claimed like the class it mimics.
    mro = manager.__class__.__mro__
    best: Optional[CacheSpec] = None
    best_depth = len(mro)
    for spec in _SPECS.values():
        for path in spec.managers:
            cls = _resolve(path)
            if cls is None or cls not in mro:
                continue
            depth = mro.index(cls)
            if depth < best_depth:
                best, best_depth = spec, depth
    return best if best is not None else spec_named(DEFAULT_SPEC)


def specs_in(page_table: Optional[PageTable]) -> List[CacheSpec]:
    """The distinct specs of a page table's spaces, in first-use order; unknown specs raise."""
    if page_table is None:
        return []
    out: List[CacheSpec] = []
    for space in page_table.spaces:
        spec = spec_for(space)
        if spec not in out:
            out.append(spec)
    return out
