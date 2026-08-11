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
"""Stable identity and registration for MoE implementations.

A Python class name cannot serve as the identity of an implementation: today a
single ``TRTLLMGenFusedMoE`` class covers eleven distinct quant x provider
combinations, so "which implementation ran" is not answerable from the class
alone. :class:`MoEImplId` makes that identity explicit, serializable and
round-trippable, which is what a persisted tuning result must key on.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, TypeVar

from .impl_contract import MoEInputRequirement, MoEStaticCapability
from .interface import MoESchedulerKind

_FIELD_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
_SEP = "."

# Registration is a decorator, so it must hand the CONCRETE class back. A bare
# ``type`` return would erase the decorated impl down to ``type[Any]``.
T = TypeVar("T")


@dataclass(frozen=True)
class MoEImplId:
    """Serializable identity of ONE MoE implementation.

    Four fields joined by ``'.'``. Each field may contain ``'_'`` internally,
    which is exactly why ``'.'`` is the separator: quant values such as
    ``w4a8_mxfp4_mxfp8`` already carry underscores, so an all-underscore
    encoding would not be parseable back into four fields.

    The four fields together must pick out exactly ONE kernel. The first three
    narrow the space; ``kernel_name`` carries the final disambiguation, so it
    names the specific kernel and not a category it belongs to. Naming a family
    instead -- ``blockscale`` when two block-scale kernels differ in, say,
    their tiling -- makes both kernels compute the same id, and
    :class:`MoEImplRegistry` then rejects the second as a duplicate, so it
    cannot be registered at all. When two kernels would otherwise collide,
    extend ``kernel_name`` with whatever actually separates them rather than
    overloading one of the other three fields.
    """

    provider: str  # maintaining entity: trtllm_native | flashinfer | deepgemm
    technique: str  # impl tech: cutlass | cutedsl | cuda_cpp | trtllm_gen
    quant: str  # weight/act format: nvfp4 | w4a8_mxfp4_mxfp8 | bf16
    # Where uniqueness of the whole id is won or lost: anything the three
    # fields above leave ambiguous has to be spelled out here, or two distinct
    # kernels collide on one id and the registry refuses to accept the second.
    kernel_name: str  # densegemm | blockscale | blockscale_splitk

    def __post_init__(self) -> None:
        for name in ("provider", "technique", "quant", "kernel_name"):
            value = getattr(self, name)
            if not _FIELD_RE.match(value):
                raise ValueError(f"MoEImplId.{name}={value!r} must match {_FIELD_RE.pattern}")

    def canonical(self) -> str:
        return _SEP.join((self.provider, self.technique, self.quant, self.kernel_name))

    @classmethod
    def parse(cls, text: str) -> "MoEImplId":
        parts = text.split(_SEP)
        if len(parts) != 4:
            raise ValueError(f"expected 4 {_SEP!r}-separated fields, got {len(parts)}: {text!r}")
        # Going through the constructor is what checks the segments: the count
        # check above says nothing about their contents, and parse() is the
        # untrusted door -- user YAML and persisted tuning results come in here.
        return cls(*parts)

    def __str__(self) -> str:
        return self.canonical()


@dataclass(frozen=True)
class MoEImplDescriptor:
    """Declaration-time metadata attached to one implementation class.

    Everything here must be knowable WITHOUT constructing the impl and without
    touching the GPU. That property is what lets offline tuning enumerate
    candidates on a machine that has no such device.
    """

    identity: MoEImplId  # exactly one id per registered class
    scheduler_kind: MoESchedulerKind  # picks ExternalComm / FusedComm
    capabilities: MoEStaticCapability = field(default_factory=lambda: MoEStaticCapability())
    input_requirement: MoEInputRequirement = field(default_factory=lambda: MoEInputRequirement())
    doc: str = ""  # human-facing only, never parsed

    @property
    def impl_id(self) -> str:
        return self.identity.canonical()


class MoEImplRegistry:
    """MoEImplId -> implementation class. Duplicate identity is a hard error."""

    def __init__(self) -> None:
        self._store: dict[MoEImplId, type] = {}

    def register(self, cls: type[T]) -> type[T]:
        descriptor = getattr(cls, "descriptor", None)
        if not isinstance(descriptor, MoEImplDescriptor):
            raise TypeError(f"{cls.__name__} must declare a MoEImplDescriptor class attribute")
        identity = descriptor.identity
        previous = self._store.get(identity)
        if previous is not None:
            # Fires at import time, not mid-run: two classes claiming the same
            # id is always a bug, never a thing to resolve by picking one.
            raise ValueError(
                f"duplicate MoEImplId {identity.canonical()}: {previous.__name__} vs {cls.__name__}"
            )
        self._store[identity] = cls
        return cls

    def lookup(self, identity: MoEImplId) -> Optional[type]:
        return self._store.get(identity)

    def __len__(self) -> int:
        return len(self._store)


# Ships EMPTY. Entries are added one per impl class as backends migrate; until
# then every concrete-impl request fails hard rather than falling back.
MOE_IMPL_REGISTRY = MoEImplRegistry()


def register_moe_impl(cls: type[T]) -> type[T]:
    """Class decorator registering an impl at class-definition time.

    Registration is an import side effect and never happens during
    ``create_moe`` -- ``create_moe`` only looks up.
    """
    return MOE_IMPL_REGISTRY.register(cls)
