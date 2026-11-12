# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feature state attached to attention metadata.

An ``AttentionMetadata`` holds the batch layout every attention backend needs:
request counts, sequence and KV lengths, the paged KV cache and the workspace.
Everything a feature, a sparse algorithm or a model adds on top lives in a
``MetadataExtension`` attached to it. The metadata drives each extension
through the hooks below at fixed points of its lifecycle, and callers reach an
extension's state with ``metadata.ext(T)`` or ``metadata.find_ext(T)``.

Every extension declares the public state it exposes in ``fields`` and the
public methods in ``methods``. Names are unique: no two extension classes
outside one inheritance line declare the same name (checked when a class is
defined), and an extension's names may not shadow its metadata's own
attributes or another attached extension's (checked when it is attached).
Lifecycle hooks are reserved and never declared.

Lifecycle, driven by the metadata:

- The constructor (``__init__(metadata, *, source=None, **config)``) calls
  ``super().__init__`` first, which sets every ``init_fields`` entry from the
  routed creation arguments or its declared default; it then sets the
  extension's other state and allocates its persistent buffers from the
  metadata's pool, and may read the extensions constructed before it. Every
  CUDA graph, cross-attention or draft metadata is constructed anew from the
  same configuration; an attribute the extension declares in ``shared`` comes
  from ``shared_or(name, allocate)``, the source's object on a copy.
- ``prepare()`` runs once per step after the core derived, in dependency
  order, and every forward of the metadata follows its own ``prepare``. An
  extension derives its own state there and corrects the core's only through
  core methods. When the core applied ``kv_lens_offsets`` (corrections known
  only on device) or an extension prepared earlier corrected them on device
  (Helix's rank-local lengths of speculative verify groups), the device KV
  lengths differ from the host values, and an extension derives its
  length-dependent device state from the device lengths.

Values the core or the engine reads from whichever extension provides them
(``kv_cache_pools``, ``full_block_table_width``, ``adp_input_token_count``,
``adp_token_counts``,
``token_major_gen_view``, ``mla_scheduler_buffers``)
are ordinary declared fields and methods, read with ``metadata.find(name,
default)``.

Opt-in members, called only by ``DraftMetadata`` on the extensions of a
one-model draft metadata (``md.draft``):

- ``adopt_target_state(source)`` takes the target's prepared state at the
  start of a draft loop (by the ``draft_*`` lists; an override calls
  ``super()`` first and fixes up what the lists cannot express).
- ``refresh_from_draft_lengths()`` re-derives the state that depends on the
  draft's lengths after the draft loop changed them in place, and resets the
  state the previous draft forward left. ``DraftMetadata.refresh_draft()``
  calls it before a draft forward, only when the draft's lengths changed.

Length-derived state an extension computes lazily is invalidated by the
extension itself, in its ``prepare`` and ``refresh_from_draft_lengths``.
"""

from __future__ import annotations

import copy
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import torch

if TYPE_CHECKING:
    from .interface import AttentionMetadata

E = TypeVar("E", bound="MetadataExtension")

# Containers a draft copies when it adopts its target's state, so that
# clearing or filling its own never changes the target's.
_COPIED_CONTAINERS = (dict, set)
# Dtypes torch._foreach_copy_ has no kernel for; copied one tensor at a time.
_NO_FOREACH_COPY_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})
# Attributes the adoption itself keeps on a draft's extension.
_ADOPTION_BOOKKEEPING = frozenset({"metadata", "_draft_copy_plan", "_draft_adopted_names"})
# The attribute a copy's extension keeps its source in while it is constructed.
_COPY_SOURCE = "_copy_source"


# Every name an extension class declares in ``fields`` / ``methods``, mapped to
# the class that declared it first.
_DECLARED_NAMES: Dict[str, type] = {}
# Every metadata constructor argument, mapped to the extension class that takes it.
_INIT_FIELD_OWNERS: Dict[str, type] = {}


class MetadataExtension:
    """State that one feature, sparse algorithm or model attaches to a metadata.

    Subclasses keep their state as attributes and override only the hooks they
    need; every hook defaults to doing nothing. ``self.metadata`` is the
    metadata the extension is attached to; read the core through it and never
    assign the core's private fields.
    """

    #: Extensions that must be attached, and prepared, before this one. They
    #: are attached automatically when this one is.
    requires: ClassVar[Tuple[Type["MetadataExtension"], ...]] = ()
    #: Tensor attributes produced and consumed inside a captured graph; the
    #: strict CUDA-graph buffer check skips them.
    graph_temporary_attrs: ClassVar[FrozenSet[str]] = frozenset()
    #: Public state this class adds; subclasses inherit their bases' fields.
    fields: ClassVar[FrozenSet[str]] = frozenset()
    #: Public methods this class adds, other than lifecycle hooks.
    methods: ClassVar[FrozenSet[str]] = frozenset()
    #: Whether this is a metadata family's core state; exactly one core is
    #: attached to each metadata, before every other extension.
    is_core: ClassVar[bool] = False
    #: Metadata constructor arguments this class takes, each mapped to its
    #: default; each is one of its declared fields. A mutable default is
    #: copied per extension.
    init_fields: ClassVar[Dict[str, Any]] = {}
    #: Whether this extension is constructed after every other one, because
    #: its constructor needs the complete metadata.
    constructed_last: ClassVar[bool] = False
    #: Attributes a copy of the metadata (a CUDA graph or cross-attention
    #: metadata) takes by reference from its source's extension instead of
    #: allocating them: kernel workspaces, buffers written on the source each
    #: step and caches every copy must see.
    shared: ClassVar[FrozenSet[str]] = frozenset()
    #: Whether a draft metadata takes this extension's prepared state from its
    #: target's (``adopt_target_state``) instead of deriving it again. A draft
    #: reuses the target's state only when every extension attached to it does.
    adopts_target_state: ClassVar[bool] = False
    #: Whether a draft takes this extension's state from its target's at every
    #: draft-loop start, also a draft that derives the rest itself: state that
    #: depends on neither the KV cache nor other extensions.
    adopts_target_state_always: ClassVar[bool] = False
    #: Attributes a draft adopting the target's state binds to the target's:
    #: the prepared state it reads but never writes.
    draft_bound: ClassVar[FrozenSet[str]] = frozenset()
    #: Tensors a draft loop writes in place. A draft adopting the target's
    #: state keeps its own and copies the target's contents into them.
    draft_written: ClassVar[FrozenSet[str]] = frozenset()
    #: Attributes a draft adopting the target's state keeps as they are: its
    #: configuration, KV cache managers, workspaces and lazily built buffers.
    #: Anything in none of the three lists is also kept; the contract tests
    #: require every attribute of an adopting extension to be listed.
    draft_private: ClassVar[FrozenSet[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        own = cls._own_declared_names()
        reserved = own & reserved_extension_names()
        if reserved:
            raise TypeError(f"{cls.__name__} declares reserved names {sorted(reserved)}")
        overlap = cls.__dict__.get("fields", frozenset()) & cls.__dict__.get("methods", frozenset())
        if overlap:
            raise TypeError(f"{cls.__name__} declares {sorted(overlap)} as both field and method")
        for name in cls.__dict__.get("init_fields", ()):
            if name not in cls.declared_fields():
                raise TypeError(f"{cls.__name__} takes {name!r} but does not declare it a field")
            _INIT_FIELD_OWNERS.setdefault(name, cls)
        if cls.is_core:
            # A core is its family's own state, checked against the extensions
            # attached with it rather than against every extension class.
            return
        for name in own:
            owner = _DECLARED_NAMES.get(name)
            if owner is None or issubclass(cls, owner) or issubclass(owner, cls):
                continue
            # A reloaded module redefines the same class.
            if (owner.__module__, owner.__qualname__) != (cls.__module__, cls.__qualname__):
                raise TypeError(
                    f"{cls.__name__} declares {name!r}, which {owner.__name__} already declares"
                )
        for name in own:
            owner = _DECLARED_NAMES.get(name)
            if owner is None or not issubclass(cls, owner):
                _DECLARED_NAMES[name] = cls

    @classmethod
    def _own_declared_names(cls) -> FrozenSet[str]:
        return cls.__dict__.get("fields", frozenset()) | cls.__dict__.get("methods", frozenset())

    @classmethod
    def declared_fields(cls) -> FrozenSet[str]:
        """Every field this class and its bases declare."""
        return frozenset().union(
            *(vars(klass).get("fields", ()) for klass in cls.__mro__ if klass is not object)
        )

    @classmethod
    def declared_methods(cls) -> FrozenSet[str]:
        """Every public method this class and its bases declare."""
        return frozenset().union(
            *(vars(klass).get("methods", ()) for klass in cls.__mro__ if klass is not object)
        )

    @classmethod
    def declared_names(cls) -> FrozenSet[str]:
        return cls.declared_fields() | cls.declared_methods()

    @classmethod
    def declared_init_fields(cls) -> Tuple[str, ...]:
        """Every constructor argument this class and its bases take, bases first."""
        return tuple(cls.init_field_defaults())

    @classmethod
    def init_field_defaults(cls) -> Dict[str, Any]:
        """Every constructor argument this class and its bases take, with its
        default, bases first; a subclass's default replaces its base's."""
        defaults: Dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            defaults.update(vars(klass).get("init_fields", {}))
        return defaults

    def __init__(
        self,
        metadata: "AttentionMetadata",
        *,
        source: Optional["MetadataExtension"] = None,
        **config: Any,
    ) -> None:
        """Construct this extension of ``metadata``.

        Sets every ``init_fields`` entry of the class and its bases from
        ``config`` (the metadata constructor arguments routed to this
        extension) or from its declared default. A subclass calls this first,
        then sets its other state and allocates its buffers from the
        metadata's pool (``metadata.cuda_graph_buffers``, graph-stable on a
        CUDA graph metadata), so it allocates after its base. ``source`` is
        the same extension of the metadata this one's is a copy of; see
        ``shared_or``.
        """
        self.metadata = metadata
        defaults = type(self).init_field_defaults()
        unknown = set(config) - set(defaults)
        if unknown:
            raise TypeError(f"{type(self).__name__} takes no {sorted(unknown)}")
        if source is not None:
            self.__dict__[_COPY_SOURCE] = source
        for name, default in defaults.items():
            setattr(self, name, config[name] if name in config else copy.copy(default))

    def shared_or(self, name: str, allocate: Callable[[], Any]) -> Any:
        """The source's ``name`` on a copy of a metadata, else ``allocate()``.

        For the attributes the class declares in ``shared``, set from its
        constructor.
        """
        if name not in type(self).declared_shared():
            raise AttributeError(f"{type(self).__name__} does not declare {name!r} shared")
        source = self.__dict__.get(_COPY_SOURCE)
        if source is not None and name in vars(source):
            return vars(source)[name]
        return allocate()

    @classmethod
    def declared_shared(cls) -> FrozenSet[str]:
        """Every attribute this class and its bases share with copies."""
        return frozenset().union(
            *(vars(klass).get("shared", ()) for klass in cls.__mro__ if klass is not object)
        )

    @classmethod
    def _declared_union(cls, name: str) -> FrozenSet[str]:
        return frozenset().union(
            *(vars(klass).get(name, ()) for klass in cls.__mro__ if klass is not object)
        )

    @classmethod
    def _adoption_names(cls) -> Tuple[FrozenSet[str], Tuple[str, ...]]:
        """The attributes a draft binds to the target's, and the written ones."""
        names = vars(cls).get("_adoption_names_cache")
        if names is None:
            names = (
                cls._declared_union("draft_bound"),
                tuple(sorted(cls._declared_union("draft_written"))),
            )
            cls._adoption_names_cache = names
        return names

    @classmethod
    def draft_unlisted(cls, names) -> FrozenSet[str]:
        """The attribute ``names`` an adopting extension lists in none of its
        draft lists (bookkeeping of the adoption itself excluded)."""
        listed = (
            cls._declared_union("draft_bound")
            | cls._declared_union("draft_written")
            | cls._declared_union("draft_private")
            | _ADOPTION_BOOKKEEPING
        )
        return frozenset(names) - listed

    def adopt_target_state(self, source: "MetadataExtension") -> None:
        """Take ``source``'s prepared state as this draft's.

        The ``draft_bound`` attributes are bound to the source's, with
        dictionaries and sets copied. The ``draft_written`` tensors, and the
        tensors of ``draft_written`` dictionaries, keep their own storage and
        get the source's contents on device (into a new buffer of the
        draft's when the shapes differ).
        Everything else stays as it is.
        """
        bound, written = type(self)._adoption_names()
        state = vars(self)
        source_state = vars(source)
        # The adopted names change only when the source gains an attribute.
        names = state.get("_draft_adopted_names")
        if names is None or names[0] != len(source_state):
            adopted = tuple(name for name in source_state if name in bound)
            containers = tuple(
                name for name in adopted if type(source_state[name]) in _COPIED_CONTAINERS
            )
            if len(adopted) > 1:
                getter = operator.itemgetter(*adopted)
            else:
                getter = lambda values: tuple(values[name] for name in adopted)  # noqa: E731
            names = (len(source_state), adopted, getter, containers)
            state["_draft_adopted_names"] = names
        _, adopted, getter, containers = names
        state.update(zip(adopted, getter(source_state)))
        for name in containers:
            state[name] = state[name].copy()
        sources = tuple(source_state.get(name) for name in written)
        mine = tuple(state.get(name) for name in written)
        plan = state.get("_draft_copy_plan")
        if (
            plan is None
            or any(a is not b for a, b in zip(sources, plan[0]))
            or any(a is not b for a, b in zip(mine, plan[1]))
        ):
            plan = self._plan_draft_copies(written, sources, mine)
        for destinations, origins in plan[2]:
            if destinations[0].dtype in _NO_FOREACH_COPY_DTYPES:
                for destination, origin in zip(destinations, origins):
                    destination.copy_(origin, non_blocking=True)
            else:
                torch._foreach_copy_(destinations, origins, non_blocking=True)

    def _plan_draft_copies(self, written, sources, mine):
        """Group the written tensors into one multi-tensor copy per device and
        dtype, first allocating the draft's own where it has none of the
        source's shape.

        A written dictionary of tensors is copied entry by entry into a
        dictionary of the draft's own."""
        state = vars(self)
        groups: Dict[Tuple[Any, Any], Tuple[List[Any], List[Any]]] = {}

        def copyable(own, value) -> bool:
            return (
                isinstance(own, torch.Tensor)
                and own is not value
                and own.shape == value.shape
                and own.dtype == value.dtype
                and own.device == value.device
            )

        def group(own, value) -> None:
            destinations, origins = groups.setdefault((own.device, own.dtype), ([], []))
            destinations.append(own)
            origins.append(value)

        for name, value, own in zip(written, sources, mine):
            if isinstance(value, dict):
                own_entries = own if isinstance(own, dict) and own is not value else {}
                entries = {}
                for key, entry in value.items():
                    own_entry = own_entries.get(key)
                    if not isinstance(entry, torch.Tensor):
                        entries[key] = entry
                    else:
                        if not copyable(own_entry, entry):
                            own_entry = torch.empty_like(entry)
                        entries[key] = own_entry
                        group(own_entry, entry)
                if (
                    own_entries is not own
                    or entries.keys() != own_entries.keys()
                    or any(entries[key] is not own_entries[key] for key in entries)
                ):
                    state[name] = entries
            elif not isinstance(value, torch.Tensor):
                state[name] = value
            else:
                if not copyable(own, value):
                    own = state[name] = torch.empty_like(value)
                group(own, value)
        plan = (
            sources,
            tuple(state.get(name) for name in written),
            tuple(groups.values()),
        )
        state["_draft_copy_plan"] = plan
        return plan

    def creation_config(self) -> Dict[str, Any]:
        """This extension's creation arguments as they stand, for a copy of its metadata."""
        return {name: self.creation_value(name) for name in type(self).declared_init_fields()}

    def creation_value(self, name: str) -> Any:
        """The current value of creation argument ``name``; see ``creation_config``."""
        return getattr(self, name)

    def prepare(self) -> None:
        """Prepare this extension's per-step state after the core."""

    def refresh_from_draft_lengths(self) -> None:
        """Re-derive this draft extension's state from its changed lengths.

        Called by ``DraftMetadata.refresh_draft()`` on the extensions of
        ``md.draft``, in dependency order, after the draft loop changed the
        draft's lengths in place and before the next draft forward. Refresh
        the state derived from the lengths and reset the previous forward's
        state; an override calls ``super()``.
        """


def init_field_owner(name: str) -> Optional[type]:
    """The extension class that takes the metadata constructor argument ``name``."""
    return _INIT_FIELD_OWNERS.get(name)


def reserved_extension_names() -> FrozenSet[str]:
    """Names no extension may declare: its own lifecycle and bookkeeping members."""
    return frozenset(name for name in vars(MetadataExtension) if not name.startswith("__")) | {
        "metadata"
    }


def check_extension_names(
    metadata: "AttentionMetadata",
    cls: Type[MetadataExtension],
    attached: Iterable[MetadataExtension],
) -> None:
    """Raise when ``cls`` declares a name its metadata or an attached extension owns.

    A core provides its metadata family's state, so it is checked against the
    other extensions only.
    """
    names = cls.declared_names()
    shadowed = set() if cls.is_core else names & metadata.own_attribute_names()
    if shadowed:
        raise ValueError(
            f"{cls.__name__} declares {sorted(shadowed)}, which "
            f"{type(metadata).__name__} already defines"
        )
    for other in attached:
        if issubclass(cls, type(other)) or isinstance(other, cls):
            continue
        clash = names & type(other).declared_names()
        if clash:
            raise ValueError(
                f"{cls.__name__} and {type(other).__name__} both declare {sorted(clash)}"
            )


def resolve_extension_order(
    types: Iterable[Type[MetadataExtension]],
) -> List[Type[MetadataExtension]]:
    """Order extension types so every requirement precedes its dependents.

    Requirements that are not listed are added. When a base extension and its
    subclass are both present, the subclass replaces the base at the base's
    dependency position. Listing order is otherwise preserved.
    """
    seed = list(types)
    all_types: List[Type[MetadataExtension]] = []
    seen: set = set()

    def collect(cls: Type[MetadataExtension]) -> None:
        if cls in seen:
            return
        seen.add(cls)
        for required in cls.requires:
            collect(required)
        all_types.append(cls)

    for cls in seed:
        collect(cls)

    replacement: Dict[Type[MetadataExtension], Type[MetadataExtension]] = {}
    for cls in all_types:
        derived = [other for other in all_types if issubclass(other, cls)]
        most_derived = [
            cand
            for cand in derived
            if all(not issubclass(other, cand) or other is cand for other in derived)
        ]
        if len(most_derived) > 1:
            raise TypeError(
                f"Ambiguous subclasses for {cls.__name__}: "
                f"{[cand.__name__ for cand in most_derived]}"
            )
        replacement[cls] = most_derived[0]

    ordered: List[Type[MetadataExtension]] = []
    visiting: set = set()

    def visit(cls: Type[MetadataExtension]) -> None:
        cls = replacement[cls]
        if cls in ordered:
            return
        if cls in visiting:
            raise ValueError(f"Metadata extension requirements form a cycle at {cls.__name__}.")
        visiting.add(cls)
        for required in cls.requires:
            target = replacement[required]
            if target is not cls:
                visit(target)
        visiting.discard(cls)
        ordered.append(cls)

    for cls in seed:
        visit(cls)
    return ordered


class ExtensionRegistry:
    """The extensions attached to one metadata, in dependency order."""

    def __init__(self, owner: Optional["AttentionMetadata"] = None) -> None:
        self.owner = owner
        self._by_type: Dict[Type[MetadataExtension], MetadataExtension] = {}

    def __iter__(self):
        return iter(self._by_type.values())

    def __len__(self) -> int:
        return len(self._by_type)

    def __contains__(self, cls: Type[MetadataExtension]) -> bool:
        return self.find(cls) is not None

    def types(self) -> List[Type[MetadataExtension]]:
        return list(self._by_type)

    def core(self) -> Optional[MetadataExtension]:
        """The attached core, which is always the first extension, or None."""
        first = next(iter(self._by_type.values()), None)
        return first if first is not None and first.is_core else None

    def attach_core(self, core: MetadataExtension) -> None:
        """Attach the metadata family's core; it must come before every other extension."""
        if not core.is_core:
            raise TypeError(f"{type(core).__name__} is not a core metadata")
        if self._by_type:
            raise ValueError(
                f"The core {type(core).__name__} must be attached before "
                f"{[t.__name__ for t in self._by_type]}"
            )
        self._by_type[type(core)] = core

    def _require_core(self, cls: Type[MetadataExtension]) -> None:
        if cls.is_core:
            raise ValueError(f"Attach the core {cls.__name__} with attach_core()")
        if self.core() is None:
            raise ValueError(f"Attach the core metadata before {cls.__name__}")

    def find(self, cls: Type[E]) -> Optional[E]:
        found = self._by_type.get(cls)
        if found is not None:
            return found
        for extension in self._by_type.values():
            if isinstance(extension, cls):
                return extension
        return None

    def attach(
        self,
        metadata: "AttentionMetadata",
        types: Sequence[Type[MetadataExtension]],
        configs: Optional[Dict[type, Dict[str, Any]]] = None,
        sources: Optional[Dict[type, MetadataExtension]] = None,
    ) -> List[MetadataExtension]:
        """Construct and attach the given types and their requirements; return the new ones.

        Each extension is constructed with its routed creation arguments
        (``configs``) and, for a copy, the same extension of the source
        metadata (``sources``), in dependency order, so its constructor can
        read the extensions attached before it.
        """
        configs = configs or {}
        sources = sources or {}
        added = []
        for cls in resolve_extension_order(types):
            if cls not in self._by_type:
                self._require_core(cls)
                check_extension_names(metadata, cls, self._by_type.values())
                # Attached before it is constructed, so its constructor reads
                # its own and the earlier extensions' fields through the
                # metadata, as every later step does.
                extension = cls.__new__(cls)
                self._by_type[cls] = extension
                metadata._compose()
                if cls in sources:
                    extension.__init__(metadata, source=sources[cls], **configs.get(cls, {}))
                else:
                    extension.__init__(metadata, **configs.get(cls, {}))
                extension.__dict__.pop(_COPY_SOURCE, None)
                added.append(extension)
        return added

    def attach_new_core(
        self,
        metadata: "AttentionMetadata",
        cls: Type[MetadataExtension],
        config: Dict[str, Any],
        source: Optional[MetadataExtension],
    ) -> MetadataExtension:
        """Construct and attach the metadata family's core, attached before it is built."""
        core = cls.__new__(cls)
        self.attach_core(core)
        metadata._compose()
        if source is not None:
            core.__init__(metadata, source=source, **config)
        else:
            core.__init__(metadata, **config)
        core.__dict__.pop(_COPY_SOURCE, None)
        return core

    def replace(self, extension: MetadataExtension) -> None:
        """Put an already-built extension in place of any of its type.

        A core replaces the attached core in place, so it stays first.
        """
        if extension.is_core:
            current = self.core()
            if current is None:
                self.attach_core(extension)
                return
            self._by_type = {
                (type(extension) if e is current else t): (extension if e is current else e)
                for t, e in self._by_type.items()
            }
            return
        if type(extension) not in self._by_type:
            self._require_core(type(extension))
        self._by_type[type(extension)] = extension

    def remove(self, cls: Type[MetadataExtension]) -> None:
        """Detach the extension of type ``cls``, if any."""
        found = self.find(cls)
        if found is not None:
            if found.is_core:
                raise ValueError(f"The core {type(found).__name__} cannot be detached")
            del self._by_type[type(found)]

    def bound_to(self, metadata: "AttentionMetadata") -> "ExtensionRegistry":
        """This registry, owned by ``metadata``.

        Returns ``self`` when ``metadata`` already owns it; otherwise a new
        registry holding a view of each extension, so a shallow
        copy of the metadata never shares the registry or its extensions with
        its source.
        """
        if self.owner is metadata:
            return self
        clone = ExtensionRegistry(metadata)
        clone._by_type = {cls: _view_of(e, metadata) for cls, e in self._by_type.items()}
        return clone

    def with_core_copied(self, metadata: "AttentionMetadata") -> "ExtensionRegistry":
        """A registry for a ``copy.copy`` of the owner: its own core, shared other extensions.

        The copy owns its core state, as it owned the core's fields before they
        moved into the core; the other extensions stay shared until the copy
        binds its own registry (``bound_to``), as before.
        """
        clone = ExtensionRegistry(self.owner)
        clone._by_type = {
            cls: (_view_of(e, metadata) if e.is_core else e) for cls, e in self._by_type.items()
        }
        return clone


def _view_of(extension: MetadataExtension, metadata: "AttentionMetadata") -> MetadataExtension:
    """A shallow view of ``extension`` bound to ``metadata`` (a ``copy.copy`` view).

    Views share every attribute with their source; a CUDA graph or
    cross-attention copy is constructed instead (see ``shared``).
    """
    view = copy.copy(extension)
    view.metadata = metadata
    return view
