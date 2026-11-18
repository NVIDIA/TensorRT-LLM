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
"""Base class, registry and switch of the feature-split attention metadata."""

from __future__ import annotations

import os
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import torch

if TYPE_CHECKING:
    from .batch_inputs import BatchInputs

ATTENTION_METADATA_ENV = "TRTLLM_ATTENTION_METADATA"

TMetadata = TypeVar("TMetadata", bound="Metadata")


def attention_metadata_mode() -> Literal["old", "new"]:
    """The attention metadata path selected by ``TRTLLM_ATTENTION_METADATA``."""
    mode = os.environ.get(ATTENTION_METADATA_ENV, "old").strip().lower()
    if mode not in ("old", "new"):
        raise ValueError(f"{ATTENTION_METADATA_ENV} must be 'old' or 'new', got {mode!r}")
    return mode


class UnsupportedMetadataConfiguration(NotImplementedError):
    """A configuration the feature-split attention metadata does not cover."""

    def __init__(self, what: str):
        super().__init__(
            f"{ATTENTION_METADATA_ENV}=new does not support {what}; "
            f"unset {ATTENTION_METADATA_ENV} to run it on the old metadata path"
        )


class _Registry:
    """The metadata built for one step: type -> instance, in prepare order."""

    __slots__ = ("by_type", "ordered", "field_owners", "private_holders", "sealed")

    def __init__(self) -> None:
        self.by_type: Dict[type, "Metadata"] = {}
        self.ordered: List["Metadata"] = []
        # Public field name -> the metadata that holds it.
        self.field_owners: Dict[str, "Metadata"] = {}
        # Underscore-prefixed name -> the metadata a read of it resolved to.
        self.private_holders: Dict[str, "Metadata"] = {}
        # Set once every metadata is constructed. Until then a second owner of
        # a field name is a declaration error.
        self.sealed = False

    def add(self, metadata: "Metadata") -> None:
        self.ordered.append(metadata)
        for cls in type(metadata).__mro__:
            if cls is Metadata:
                break
            self.by_type.setdefault(cls, metadata)
            for name in vars(cls):
                if name[0] != "_" and name not in _BASE_NAMES:
                    self.claim(name, metadata)

    def claim(self, name: str, metadata: "Metadata") -> "Metadata":
        """The owner of field ``name``: ``metadata`` unless another holds it."""
        owner = self.field_owners.setdefault(name, metadata)
        if owner is not metadata and not self.sealed:
            raise TypeError(
                f"attention metadata field {name!r} is defined by both "
                f"{type(owner).__name__} and {type(metadata).__name__}; "
                "public field names must be unique across the attached metadata"
            )
        return owner

    def holder_of(self, name: str) -> Optional["Metadata"]:
        """The metadata a read of ``name`` resolves to, or None."""
        if name[0] != "_":
            return self.field_owners.get(name)
        holder = self.private_holders.get(name)
        if holder is None:
            for metadata in self.ordered:
                if name in vars(metadata) or hasattr(type(metadata), name):
                    holder = self.private_holders[name] = metadata
                    break
        return holder


class Metadata:
    """One attention feature's metadata.

    A subclass owns its fields, creates them in ``__init__`` and fills them in
    ``prepare``; it has no other entry point. ``requires`` names the metadata
    it depends on; they are constructed and prepared first.

    The public fields of every attached metadata read and assign through any
    of them: ``metadata.field`` reaches the metadata that owns ``field``. A
    public field name therefore has one owner among the attached metadata.
    A name with a leading underscore belongs to the instance it is assigned
    on; a read of one that the instance lacks resolves to the first attached
    metadata that holds it.
    """

    requires: ClassVar[Tuple[Type["Metadata"], ...]] = ()
    # This metadata builds further metadata from the attached types, so it is
    # constructed last.
    builds_metadata: ClassVar[bool] = False

    def __init__(self, registry: _Registry, inputs: "BatchInputs") -> None:
        self._registry = registry
        # The description this metadata was constructed from; its engine
        # fields size the buffers. Every metadata holds it, so it is set
        # without claiming the name.
        object.__setattr__(self, "config", inputs)

    def __getattr__(self, name: str):
        # Reached only when this metadata does not hold ``name`` itself.
        registry = self.__dict__.get("_registry")
        if registry is not None and not name.startswith("__"):
            holder = registry.holder_of(name)
            if holder is not None and holder is not self:
                return getattr(holder, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}, "
            "and no attached metadata defines it"
        )

    def __setattr__(self, name: str, value) -> None:
        if name[0] != "_":
            owner = self._registry.claim(name, self)
            if owner is not self:
                object.__setattr__(owner, name, value)
                return
        object.__setattr__(self, name, value)

    def get_metadata(self, metadata_type: Type[TMetadata]) -> Optional[TMetadata]:
        """The attached instance of ``metadata_type`` or of a subclass, or None."""
        return self._registry.by_type.get(metadata_type)

    def prepare(self, inputs: "BatchInputs") -> None:
        """Fill this metadata's fields for the step ``inputs`` describes."""

    def prepare_all(self, inputs: "BatchInputs") -> None:
        """Prepare every attached metadata for one step, dependencies first."""
        for metadata in self._registry.ordered:
            metadata.prepare(inputs)

    def attached_types(self) -> List[Type["Metadata"]]:
        """The concrete types attached alongside this metadata, in prepare order."""
        return [type(metadata) for metadata in self._registry.ordered]

    def get_buffer(self, shape: Sequence[int], dtype: torch.dtype, name: str) -> torch.Tensor:
        """A zero-filled device buffer whose address a CUDA graph may capture."""
        # Imported here: the backend interface imports the sparse parameter modules.
        from ..backends.interface import AttentionMetadata

        buffer = AttentionMetadata.get_empty(
            self.config.cuda_graph_buffers,
            list(shape),
            dtype=dtype,
            cache_name=self.config.buffer_name_prefix + name,
            capture_graph=self.config.is_cuda_graph,
        )
        if self.config.cuda_graph_buffers is not None:
            # The CUDA graph buffer pool hands out uninitialized memory. A step
            # writes only the rows it uses, and prefix sums only past their
            # first element, so start from the zeros an unpooled buffer has.
            buffer.zero_()
        return buffer


# Names every metadata has; they are not fields of one feature.
_BASE_NAMES = frozenset(name for name in vars(Metadata) if name[0] != "_") | {"config"}


def build_metadata(metadata_types: Sequence[Type[Metadata]], inputs: "BatchInputs") -> Metadata:
    """Construct ``metadata_types`` and what they require.

    Returns:
        The instance of ``metadata_types[0]``; the fields of the others read
        through it, and ``get_metadata`` returns the others themselves.

    Raises:
        TypeError: Two of the metadata define the same public field.
    """
    registry = _Registry()

    def construct(metadata_type: Type[Metadata]) -> None:
        if metadata_type in registry.by_type:
            return
        for required_type in metadata_type.requires:
            construct(required_type)
        registry.add(metadata_type(registry, inputs))

    # Most-derived types first: a subclass stands in for the base it derives
    # from. A metadata that builds further metadata from the attached types
    # comes last.
    for metadata_type in sorted(
        metadata_types, key=lambda cls: (cls.builds_metadata, -len(cls.__mro__))
    ):
        construct(metadata_type)
    registry.sealed = True
    # Dependencies first; the sort is stable, so equals keep construction order.
    registry.ordered.sort(key=_depth)
    return registry.by_type[metadata_types[0]]


def _depth(metadata: Metadata) -> int:
    """Length of the longest ``requires`` chain below ``metadata``."""

    def depth(metadata_type: Type[Metadata]) -> int:
        return 1 + max((depth(required) for required in metadata_type.requires), default=0)

    return depth(type(metadata))
