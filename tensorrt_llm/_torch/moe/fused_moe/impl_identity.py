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
"""Stable identities, queries, and registration for leaf MoE implementations."""

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, TypeVar

from .impl_contract import MoEInputRequirement, MoEStaticCapability
from .interface import MoESchedulerKind

_FIELD_RE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
_SEP = "."
T = TypeVar("T")

# Canonical order for rendering an id. Parsing does not depend on it: tokens
# are assigned to fields by value, so a query may list its segments in any
# order.
_ID_FIELDS: Tuple[str, ...] = ("provider", "technique", "kernel_name", "quant")

# Explicit "any value here". Never required -- an omitted field is already
# unconstrained -- but accepted so that a query can be written out at full
# width, and so that MoEImplQuery.describe() round-trips back through
# MoEImplRegistry.parse_query().
_WILDCARD = "*"


def _normalize(name: str, value: str) -> str:
    """Lowercase an incoming segment and reject anything still malformed.

    Case folding happens here rather than at each call site so that a value
    typed as ``Cutlass`` and one typed as ``cutlass`` cannot become two
    different registry keys. The canonical form is always lowercase.
    """
    if not isinstance(value, str):
        raise TypeError(f"MoEImplId.{name} must be a string, got {type(value).__name__}")
    folded = value.strip().lower()
    if not _FIELD_RE.match(folded):
        raise ValueError(f"MoEImplId.{name}={value!r} must match {_FIELD_RE.pattern}")
    return folded


@dataclass(frozen=True)
class MoEImplId:
    """Exact ``provider.technique.kernel_name.quant`` implementation identity."""

    # Kernel lineage, e.g. trtllm, flashinfer, deepgemm, or marlin.
    provider: str  # trtllm | flashinfer | deepgemm | marlin | triton_kernels
    # Implementation technology, independent of provider.
    technique: str  # cutlass | cutedsl | cuda_cpp | trtllm_gen | triton | torch
    # Specific kernel name that makes the full identity unique.
    kernel_name: str  # grouped_gemm | dense_gemm | fused_moe | mega_moe | vanilla
    quant: str  # weight/act format; ``none`` when unquantized

    def __post_init__(self):
        for name in _ID_FIELDS:
            # Frozen dataclass, so normalization has to go around __setattr__.
            object.__setattr__(self, name, _normalize(name, getattr(self, name)))

    def canonical(self) -> str:
        return _SEP.join(getattr(self, name) for name in _ID_FIELDS)

    @classmethod
    def parse(cls, text: str) -> "MoEImplId":
        parts = text.split(_SEP)
        if len(parts) != len(_ID_FIELDS):
            raise ValueError(
                f"expected {len(_ID_FIELDS)} {_SEP!r}-separated fields, got {len(parts)}: {text!r}"
            )
        # Going through the constructor is what checks the segments: the count
        # check above says nothing about their contents, and parse() is the
        # untrusted door -- user YAML and persisted tuning results come in here.
        return cls(*parts)

    def __str__(self) -> str:
        return self.canonical()


@dataclass(frozen=True)
class MoEImplQuery:
    """Partial implementation identity; None leaves a field unconstrained."""

    provider: Optional[str] = None
    technique: Optional[str] = None
    kernel_name: Optional[str] = None
    quant: Optional[str] = None

    def __post_init__(self):
        for name in _ID_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _normalize(name, value))

    @property
    def is_empty(self) -> bool:
        """No constraint at all, i.e. every registered impl matches."""
        return all(getattr(self, name) is None for name in _ID_FIELDS)

    @property
    def is_exact(self) -> bool:
        """Every field pinned, so this names at most one implementation.

        The distinction drives failure semantics: an exact query that matches
        nothing is a hard error, while a partial one that matches several is
        resolved by priority.
        """
        return all(getattr(self, name) is not None for name in _ID_FIELDS)

    def as_impl_id(self) -> MoEImplId:
        """The single id this query names. Only valid when :attr:`is_exact`."""
        if not self.is_exact:
            raise ValueError(f"query {self.describe()} does not pin all {len(_ID_FIELDS)} fields")
        return MoEImplId(**{name: getattr(self, name) for name in _ID_FIELDS})

    def matches(self, identity: MoEImplId) -> bool:
        """Whether ``identity`` satisfies every field this query does pin."""
        return all(
            getattr(self, name) is None or getattr(self, name) == getattr(identity, name)
            for name in _ID_FIELDS
        )

    def describe(self) -> str:
        """Full-width rendering, ``*`` for the fields left open.

        Round-trips: :meth:`MoEImplRegistry.parse_query` accepts this back.
        """
        return _SEP.join(getattr(self, name) or _WILDCARD for name in _ID_FIELDS)

    def __str__(self):
        return self.describe()


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
    """Map unique implementation identities and query tokens to classes."""

    def __init__(self):
        self._store: Dict[MoEImplId, Type] = {}
        self._token_to_field: Dict[str, str] = {}

    def _check_tokens_disjoint(self, identity: MoEImplId) -> None:
        for name in _ID_FIELDS:
            token = getattr(identity, name)
            owner = self._token_to_field.get(token)
            if owner is not None and owner != name:
                raise ValueError(
                    f"cannot register {identity.canonical()}: token {token!r} is already a "
                    f"value of field {owner!r}, so a user writing {token!r} could mean either "
                    f"field. Rename one of them -- value sets must stay disjoint across fields."
                )

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
        self._check_tokens_disjoint(identity)
        self._store[identity] = cls
        for name in _ID_FIELDS:
            self._token_to_field[getattr(identity, name)] = name
        return cls

    def lookup(self, identity: MoEImplId) -> Optional[type]:
        return self._store.get(identity)

    def field_of(self, token: str) -> Optional[str]:
        """Which id field a bare token belongs to, or ``None`` if unknown."""
        return self._token_to_field.get(token.strip().lower())

    def known_tokens(self) -> Dict[str, str]:
        """Copy of the token vocabulary, for diagnostics and error messages."""
        return dict(self._token_to_field)

    def suggest(self, token: str) -> Tuple[str, ...]:
        """Nearest known tokens, in descending similarity, for a typo message.

        Empty when nothing is close, which includes the case that matters most
        during the migration: a registry too sparse to hold a neighbour.
        """
        return tuple(difflib.get_close_matches(token, self._token_to_field, n=3))

    def parse_query(self, text: str) -> MoEImplQuery:
        """Parse a partial implementation query written in any segment order.

        Every token locates its own field by value, so ``deepgemm.mega_moe``
        and ``mega_moe.deepgemm`` are the same request and a bare ``mega_moe``
        is a legal one. Position carries no meaning here; that is what
        :meth:`_check_tokens_disjoint` buys at registration time.
        """
        tokens = []
        for raw in text.split(_SEP):
            token = raw.strip().lower()
            if not token:
                raise ValueError(f"empty segment in MoE impl specification {text!r}")
            tokens.append(token)

        # The per-field duplicate check below bounds named tokens only. A
        # wildcard claims no field, so without this nothing would reject
        # "deepgemm.*.*.*.*".
        if len(tokens) > len(_ID_FIELDS):
            raise ValueError(
                f"MoE impl specification {text!r} has {len(tokens)} segments, more than "
                f"the {len(_ID_FIELDS)} fields {_SEP.join(_ID_FIELDS)}"
            )

        assignment: Dict[str, str] = {}
        for token in tokens:
            if token == _WILDCARD:
                continue
            name = self._token_to_field.get(token)
            if name is None:
                near = self.suggest(token)
                # Near misses while the registry has neighbours to compare
                # against; the full vocabulary only while it is small enough
                # for that to be a listing rather than noise.
                hint = (
                    f"Closest known values: {list(near)}"
                    if near
                    else f"Known tokens: {sorted(self._token_to_field)}"
                )
                raise ValueError(f"unknown MoE impl token {token!r} in {text!r}. {hint}")
            if name in assignment:
                raise ValueError(
                    f"MoE impl specification {text!r} sets field {name!r} twice: "
                    f"{assignment[name]!r} and {token!r}"
                )
            assignment[name] = token
        return MoEImplQuery(**assignment)

    def find(self, query: MoEImplQuery) -> List[Tuple[MoEImplId, Type]]:
        """Every registered impl the query matches, in registration order."""
        return [(ident, cls) for ident, cls in self._store.items() if query.matches(ident)]

    def identities(self) -> Tuple[MoEImplId, ...]:
        return tuple(self._store)

    def __len__(self):
        return len(self._store)


# Created empty; each leaf registers itself at import time as it migrates, so
# the registry is partial for as long as the migration runs. A pin that names
# nothing registered fails hard rather than falling back, which is what keeps a
# partial registry from answering with a kernel the caller did not ask for.
MOE_IMPL_REGISTRY = MoEImplRegistry()


def register_moe_impl(cls: type[T]) -> type[T]:
    """Class decorator registering an impl at class-definition time.

    Registration is an import side effect and never happens during
    ``create_moe`` -- ``create_moe`` only looks up.
    """
    return MOE_IMPL_REGISTRY.register(cls)
