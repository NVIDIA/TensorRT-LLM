# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import weakref
from collections import namedtuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, FrozenSet, Generic,
                    List, Optional, Protocol, Sequence, Tuple, Type, TypeVar,
                    Union)

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from ...attention.backends.prims_ts.vc_attention.quantization import VCAttentionOperands
    from ...model_config import ModelConfig
    from ...pyexecutor.llm_request import LlmRequest
    from ...speculative.interface import SpecMetadata
    from ...speculative.spec_tree_manager import SpecTreeManager
    from ..kernels.sage_quant import SageQuantized
    from .fmha.gqa_decode import KvAppendPlan

from tensorrt_llm._utils import get_hf_rope_theta
from tensorrt_llm.functional import (AttentionMaskType, PositionEmbeddingType,
                                     RopeEmbeddingUtils, RotaryScalingType)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ...host_staging import copy_host_to_device
from ...memory_buffer_utils import Buffers
from ...metadata import KVCacheParams
from ...pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from ...pyexecutor.kv_cache.mamba_cache_manager import BaseMambaCacheManager
from ...pyexecutor.resource_manager import KVCacheManager
from ...pyexecutor.trace_log_utils import log_tensor_size
from ...utils import get_model_extra_attrs
from .core import AttentionRuntimeFeatures  # noqa: F401  (public import path)
from .core import AttentionCoreMetadata
from .cpp_schema import cpp_field
from .extensions import (ExtensionRegistry, MetadataExtension, init_field_owner,
                         resolve_extension_order)
from .sparse.params import SparseBackendForwardArgs, SparseRuntimeParams

ExtensionT = TypeVar("ExtensionT", bound=MetadataExtension)
_NO_EXTENSIONS = ExtensionRegistry()
_ASSIGNABLE_NAMES: Dict[type, FrozenSet[str]] = {}
# Composed metadata classes keyed by (core class, attached extension types).
_COMPOSED_CLASSES: Dict[Tuple[type, Tuple[type, ...]], type] = {}


def _forwarded_field(extension_type: type, name: str) -> property:
    """A property reading and writing ``name`` on the attached ``extension_type``."""

    def get(self):
        return getattr(self.__dict__["_extensions"]._by_type[extension_type],
                       name)

    def set(self, value):
        setattr(self.__dict__["_extensions"]._by_type[extension_type], name,
                value)

    return property(get, set, doc=f"{extension_type.__name__}.{name}")


def _forwarded_method(extension_type: type, name: str) -> property:
    """A property returning the attached ``extension_type``'s bound ``name``."""

    def get(self):
        return getattr(self.__dict__["_extensions"]._by_type[extension_type],
                       name)

    return property(get, doc=f"{extension_type.__name__}.{name}()")


def _composed_class(core: type, extension_types: Tuple[type, ...]) -> type:
    """``core`` with each extension's declared fields and methods forwarded.

    One class per (core, extension types) combination, so ``isinstance``
    against the core family holds and ``copy.copy`` keeps the forwarding.
    """
    key = (core, extension_types)
    composed = _COMPOSED_CLASSES.get(key)
    if composed is None:
        namespace = {"_core_class": core, "__module__": core.__module__}
        core_names = _assignable_names(core)
        for extension_type in extension_types:
            fields = extension_type.declared_fields()
            methods = extension_type.declared_methods()
            if extension_type.is_core:
                # A metadata class that defines one of its core's names itself
                # (a test double) keeps its own definition.
                fields = {name for name in fields if not hasattr(core, name)}
                methods = {name for name in methods if not hasattr(core, name)}
            else:
                shadowed = (fields | methods) & core_names
                if shadowed:
                    raise ValueError(
                        f"{extension_type.__name__} declares {sorted(shadowed)}, "
                        f"which {core.__name__} already defines")
            for name in fields:
                namespace[name] = _forwarded_field(extension_type, name)
            for name in methods:
                namespace[name] = _forwarded_method(extension_type, name)
        names = ",".join(t.__name__ for t in extension_types)
        composed = type(f"{core.__name__}[{names}]", (core, ), namespace)
        _COMPOSED_CLASSES[key] = composed
    return composed


def _assignable_names(cls: type) -> FrozenSet[str]:
    """Attribute names an instance of ``cls`` may assign.

    Every name a class in the MRO declares (dataclass fields, annotations,
    class attributes, properties, methods) plus each class's
    ``_instance_attrs``.
    """
    names = _ASSIGNABLE_NAMES.get(cls)
    if names is None:
        collected = set()
        for klass in cls.__mro__:
            if klass is object:
                continue
            collected.update(vars(klass))
            collected.update(vars(klass).get("__annotations__", ()))
            collected.update(vars(klass).get("_instance_attrs", ()))
        names = _ASSIGNABLE_NAMES[cls] = frozenset(collected)
    return names


try:
    # Transformers v5
    from transformers.configuration_utils import ALLOWED_ATTENTION_LAYER_TYPES
except ImportError:
    # Transformers v4
    from transformers.configuration_utils import \
        ALLOWED_LAYER_TYPES as ALLOWED_ATTENTION_LAYER_TYPES


# The type of requests in qkv passed to attention
# Please keep sync with AttentionInputType in cpp/tensorrt_llm/thop/attentionOp.cpp
class AttentionInputType(IntEnum):
    mixed = 0  # contains both context and generation
    context_only = 1
    generation_only = 2


@dataclass(kw_only=True)
class BatchInputs:
    """One step's batch description, applied by ``AttentionMetadata.prepare``.

    ``q_offsets`` / ``kv_offsets`` are optional packed-token boundaries
    ``[num_seqs + 1]`` (int32); None means ``cumsum(seq_lens)``. Explicit
    offsets may leave gaps between sequences (``offsets[i + 1] - offsets[i] >=
    seq_lens[i]``); only metadata whose ``supports_explicit_offsets`` is True
    honors offsets that differ from ``cumsum(seq_lens)``.
    """

    # Per-sequence query lengths [num_seqs] on the host; None keeps the bound
    # lengths (CUDA graph metadata, whose lengths the graph runner owns).
    seq_lens: Optional[torch.Tensor]
    num_contexts: int
    request_ids: List[int]
    prompt_lens: List[int]
    kv_cache_params: Optional[KVCacheParams] = None
    beam_width: int = 1
    num_chunked_ctx_requests: int = 0
    q_offsets: Optional[torch.Tensor] = None
    kv_offsets: Optional[torch.Tensor] = None
    # A device copy of ``seq_lens`` to bind as it is instead of staging one,
    # shared with the metadata that owns it (a cross-attention sub-metadata
    # shares its decoder's query lengths).
    seq_lens_cuda: Optional[torch.Tensor] = None
    # Per-sequence KV lengths on the host when they differ from the query
    # lengths (cross attention); None keeps the bound ones.
    seq_lens_kv: Optional[torch.Tensor] = None
    # The step's inputs of the cross-attention sub-metadata, which prepares
    # its own state from them.
    cross: Optional["BatchInputs"] = None
    # The step's scheduled context requests, which extensions read in their
    # prepare, and whether context forwards may run in CUDA graphs.
    context_requests: Optional[List["LlmRequest"]] = None
    prefill_cuda_graphs: bool = False
    # Per-sequence corrections [num_seqs] (int32, device) to the KV lengths the
    # host-side ``kv_cache_params`` imply, known only on device when the step
    # is prepared: lengths a previous step's forward changed after it was
    # scheduled (accepted draft tokens, a draft rewind). Only metadata whose
    # ``supports_kv_lens_offsets`` is True applies them.
    kv_lens_offsets: Optional[torch.Tensor] = None
    # Per-token corrections [num_tokens - num_ctx_tokens] (int32, device) to
    # the positions of the generation tokens the host packed, from the same
    # source as ``kv_lens_offsets``; Helix derives its rank-local lengths from
    # the corrected positions.
    position_id_offsets: Optional[torch.Tensor] = None
    # Per-request part lengths of multi-item scoring; None when the batch
    # does not score multiple items per request.
    multi_item_part_lens: Optional[List[List[int]]] = None


class AttentionMetadata:
    """A container of the extensions that hold an attention metadata's state.

    The container holds no state of its own: its family's core
    (``core_metadata_type``, always attached first) holds the batch layout and
    the family's buffers, and every feature, sparse algorithm or model adds an
    extension. The container drives their lifecycle and forwards their
    declared fields and methods, so ``metadata.seq_lens`` reads the core's.

    Assigning a name that neither the container's classes nor an attached
    extension declares raises ``AttributeError``.
    """

    # Names a test double defined on a subclass may assign; the families keep
    # their state in their core.
    _instance_attrs: ClassVar[FrozenSet[str]] = frozenset()
    # The core extension holding this family's batch layout and state.
    core_metadata_type: ClassVar[
        Type[AttentionCoreMetadata]] = AttentionCoreMetadata
    # Whether ``prepare`` applies ``BatchInputs.kv_lens_offsets`` to the
    # device KV lengths before the extensions prepare.
    supports_kv_lens_offsets: ClassVar[bool] = False
    # Whether the kernels read ``cu_q_seqlens`` / ``cu_kv_seqlens`` as given,
    # so they may differ from cumsum(seq_lens). Metadata that derives the
    # boundaries itself rejects offsets that differ.
    supports_explicit_offsets: ClassVar[bool] = False

    # Extensions a backend's metadata class attaches to every instance; set on
    # the class AttentionBackend derives from its ``metadata_extensions``.
    backend_extension_types: ClassVar[Tuple[Type[MetadataExtension], ...]] = ()

    # The attached extensions, the core first; see extensions.py.
    _extensions: ExtensionRegistry

    def __init__(self,
                 *,
                 extension_types: Sequence[Type[MetadataExtension]] = (),
                 copy_of: Optional["AttentionMetadata"] = None,
                 **kwargs: Any) -> None:
        """Construct the core and the extensions, routing each argument to its owner.

        Every keyword argument is a field some extension takes (its
        ``init_fields``). The extensions attached are the family's defaults,
        chosen from the arguments, plus ``extension_types``. An argument whose
        owner is not attached is an error unless it is None, which means not
        given; an argument no extension takes is an error.

        Each extension's constructor allocates its buffers, in dependency
        order and the core first, so it can read the extensions constructed
        before it. ``copy_of`` makes this metadata a copy of another (see
        ``constructed_copy``): every extension takes its declared ``shared``
        attributes from that metadata's.
        """
        cls = type(self)
        core_type = cls.core_metadata_type
        # The backend's extensions follow the family's, whose state (such as
        # the speculative-decoding settings) they read when constructed.
        types = resolve_extension_order([
            *cls.default_extension_types(kwargs), *cls.backend_extension_types,
            *extension_types
        ])
        # Extensions that need the complete metadata come last.
        types = ([t for t in types if not t.constructed_last] +
                 [t for t in types if t.constructed_last])
        owners: Dict[str, type] = {}
        for owner in (core_type, *types):
            for name in owner.declared_init_fields():
                owners.setdefault(name, owner)
        routed: Dict[type, Dict[str, Any]] = {}
        for name, value in kwargs.items():
            owner = owners.get(name)
            if owner is not None:
                routed.setdefault(owner, {})[name] = value
                continue
            elsewhere = init_field_owner(name)
            if elsewhere is None:
                raise TypeError(f"{cls.__name__}() got an unexpected keyword "
                                f"argument {name!r}")
            if value is not None:
                raise TypeError(
                    f"{cls.__name__}() got {name!r}, which "
                    f"{elsewhere.__name__} takes, but does not attach it")
        missing = [
            name for name in core_type.required_init_fields
            if name not in routed.get(core_type, {})
        ]
        if missing:
            raise TypeError(f"{cls.__name__}() missing required keyword "
                            f"arguments {missing}")
        sources = ({} if copy_of is None else {
            type(e): e
            for e in copy_of._attached_extensions()
        })
        registry = ExtensionRegistry(self)
        self.__dict__["_extensions"] = registry
        registry.attach_new_core(self, core_type, routed.get(core_type, {}),
                                 sources.get(core_type))
        registry.attach(self, types, routed, sources)

    def constructed_copy(self,
                         *,
                         share: bool = True,
                         without: Sequence[Type[MetadataExtension]] = (),
                         **changes: Any) -> Self:
        """A new metadata built from this one's configuration, with ``changes``.

        Every extension attached now is attached to the copy, constructed with
        this metadata's current creation arguments (each extension's
        ``creation_config``) updated by ``changes``; the copy allocates its own
        buffers from its pool and, with ``share``, takes the ``shared``
        attributes from this metadata's extensions. Extensions of the types
        in ``without`` are left out. For CUDA graph, cross-attention and draft
        metadata.
        """
        config: Dict[str, Any] = {}
        for extension in self._attached_extensions():
            if not isinstance(extension, tuple(without)):
                config.update(extension.creation_config())
        config.update(changes)
        types = [
            extension_type
            for extension_type in self._attached_extensions().types()
            if not extension_type.is_core
            and not issubclass(extension_type, tuple(without))
        ]
        return self.core_class()(extension_types=types,
                                 copy_of=self if share else None,
                                 **config)

    @classmethod
    def default_extension_types(
            cls, settings: Dict[str, Any]) -> List[Type[MetadataExtension]]:
        """Extensions this family attaches, chosen from the constructor arguments."""
        from .features import CrossAttentionMetadata, DraftMetadata
        types: List[Type[MetadataExtension]] = [DraftMetadata]
        if isinstance(settings.get("kv_cache_manager"), BaseMambaCacheManager):
            from .features.recurrent import RecurrentMetadata
            types.append(RecurrentMetadata)
        if (settings.get("cross") is not None
                or settings.get("seq_lens_kv") is not None):
            types.append(CrossAttentionMetadata)
        return types

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in _assignable_names(type(self)):
            if self._routes_to_core(name):
                # A metadata built without __init__ (as some tests do) gets
                # its core first; the composed class then forwards the write.
                self._ensure_core()
                setattr(self, name, value)
                return
            raise AttributeError(
                f"{type(self).__name__} declares no attribute {name!r}; keep "
                "feature or model state in a MetadataExtension")
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        # Reached only when normal lookup fails: route a core field of a
        # metadata built without __init__ (as some tests do) to its core.
        if self._routes_to_core(name):
            self._ensure_core()
            return getattr(self, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def _routes_to_core(self, name: str) -> bool:
        if name.startswith("__"):
            return False
        registry = self.__dict__.get("_extensions")
        if registry is not None and registry.core() is not None:
            return False
        return name in type(self).core_metadata_type.declared_names()

    def __copy__(self) -> Self:
        """A shallow copy that owns a copy of the core state.

        The other extensions stay shared until the copy binds its own registry
        (``bound_to``), as with any ``copy.copy`` view of the metadata.
        """
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        registry = self.__dict__.get("_extensions")
        if registry is not None:
            clone.__dict__["_extensions"] = registry.with_core_copied(clone)
        return clone

    def core(self) -> AttentionCoreMetadata:
        """This metadata's core: its batch layout and family state."""
        return self._ensure_core()

    def _ensure_core(self) -> AttentionCoreMetadata:
        """Bind this metadata's registry and attach its core first, if missing."""
        registry = self.__dict__.get("_extensions")
        if registry is None or registry.owner is not self:
            registry = (registry or _NO_EXTENSIONS).bound_to(self)
            self.__dict__["_extensions"] = registry
        core = registry.core()
        if core is None:
            # A metadata made without __init__ (as some tests do) gets its
            # family's core with the default configuration: no capacities.
            core = registry.attach_new_core(self,
                                            type(self).core_metadata_type, {},
                                            None)
        return core

    def get_adp_token_counts(self) -> List[int]:
        """Local execution-stage counts carried by the input count exchange.

        The first stage is the input (``num_tokens``, unless an extension
        declares ``adp_input_token_count``); each extension may append its own.
        """
        input_tokens = self.find("adp_input_token_count", None)
        if input_tokens is None:
            input_tokens = self.num_tokens
        return [input_tokens, *(self.find("adp_token_counts", None) or ())]

    def set_adp_token_counts(self, counts: List[List[int]]) -> List[int]:
        """Save stage counts and return the input-stage counts for padding."""
        if len(counts) > 1:
            self.find("receive_adp_token_counts")(counts[1:], counts[0])
        return counts[0]

    def find(self, name: str, default: Any = None) -> Any:
        """An attached extension's declared field or bound method, or ``default``.

        For values the core or the engine reads from whichever extension
        provides them, such as ``kv_cache_pools`` or ``full_block_table_width``.
        """
        for extension in self._attached_extensions():
            declared = type(extension)
            if name in declared.declared_fields(
            ) or name in declared.declared_methods():
                return getattr(extension, name)
        return default

    @classmethod
    def core_class(cls) -> type:
        """The metadata family class, without the forwarding of a composed class."""
        return cls.__dict__.get("_core_class", cls)

    def _compose(self) -> None:
        """Switch to the composed class of the attached extensions.

        The composed class forwards each extension's declared fields and
        methods, so callers read ``metadata.field`` instead of
        ``metadata.ext(T).field``.
        """
        extension_types = tuple(self._attached_extensions().types())
        object.__setattr__(self, "__class__",
                           _composed_class(self.core_class(), extension_types))

    def own_attribute_names(self) -> FrozenSet[str]:
        """Names this metadata's family defines; extensions may not shadow them."""
        return _assignable_names(self.core_class())

    def _attached_extensions(self) -> ExtensionRegistry:
        # Metadata built without __init__ (as some tests do) has no registry.
        return self.__dict__.get("_extensions", _NO_EXTENSIONS)

    def ext(self, cls: Type[ExtensionT]) -> ExtensionT:
        """The attached extension of type ``cls``; raises when none is."""
        found = self._attached_extensions().find(cls)
        if found is None:
            raise LookupError(
                f"{type(self).__name__} has no {cls.__name__} extension attached."
            )
        return found

    def find_ext(self, cls: Type[ExtensionT]) -> Optional[ExtensionT]:
        """The attached extension of type ``cls``, or None."""
        return self._attached_extensions().find(cls)

    def add_ext(self, cls: Type[ExtensionT]) -> ExtensionT:
        """Construct and attach ``cls`` and its requirements if absent; return the extension."""
        self._ensure_core()
        self._extensions.attach(self, (cls, ))
        return self.ext(cls)

    def replace_ext(self, extension: MetadataExtension) -> None:
        """Attach a built extension in place of any of its type.

        For derived views made by ``copy.copy``: binds this metadata to its own
        registry first, so the source keeps its extensions.
        """
        assert extension.metadata is self, "the extension must be built for this metadata"
        if not extension.is_core:
            self._ensure_core()
        elif self.__dict__.get(
                "_extensions") is None or self._extensions.owner is not self:
            self._extensions = self._attached_extensions().bound_to(self)
        self._extensions.replace(extension)
        self._compose()

    def remove_ext(self, cls: Type[MetadataExtension]) -> None:
        """Detach the extension of type ``cls`` from a derived view.

        Like ``replace_ext``, binds this metadata to its own registry first.
        """
        self._ensure_core()
        self._extensions.remove(cls)
        self._compose()

    @property
    def extensions(self) -> List[MetadataExtension]:
        """The attached extensions in dependency order."""
        return list(self._attached_extensions())

    def get_buffer(self, shape: Sequence[int], dtype: torch.dtype,
                   name: str) -> torch.Tensor:
        """A persistent buffer from this metadata's pool, graph-stable under capture."""
        return self.get_empty(self.cuda_graph_buffers,
                              list(shape),
                              dtype=dtype,
                              cache_name=name,
                              capture_graph=self.is_cuda_graph)

    def stage_to_device(self, key: Any, destination: torch.Tensor,
                        value: torch.Tensor) -> None:
        """Copy a host tensor into ``destination`` through pinned staging."""
        copy_host_to_device(self.core()._host_staging, key, destination, value)

    def adopts_target_state(self) -> bool:
        """Whether every attached extension takes a draft's state from its target's."""
        return all(
            type(extension).adopts_target_state
            for extension in self._attached_extensions())

    def check_lengths_current(self) -> None:
        """Raise when the lengths changed in place without a refresh.

        The attention forward calls it: state derived from the lengths would
        be stale.
        """
        if self.lengths_changed:
            raise RuntimeError(
                f"{type(self).__name__}'s lengths changed in place since its "
                "last prepare or draft refresh; refresh the draft before its "
                "forward")

    @property
    def has_cross_sub_metadata(self) -> bool:
        """Whether this metadata carries a cross-attention sub-metadata."""
        from .features import CrossAttentionMetadata
        cross = self.find_ext(CrossAttentionMetadata)
        return cross is not None and cross.has_sub_metadata

    def prepare(self,
                inputs: Optional[BatchInputs] = None,
                *,
                context_requests: Optional[List["LlmRequest"]] = None,
                prefill_cuda_graphs: bool = False) -> None:
        """Prepare this step's state before the model forward.

        ``inputs`` replaces the batch description; without it the fields
        already set are used. ``inputs.cross`` prepares the cross-attention
        sub-metadata after this metadata. ``context_requests`` (or
        ``inputs.context_requests``) are this step's scheduled context requests,
        which extensions read in their own ``prepare``.

        The core derives first; every other extension then prepares in
        dependency order and may correct the core's state only through core
        methods.
        """
        core = self.core()
        if inputs is not None:
            if (inputs.kv_lens_offsets is not None
                    and not self.supports_kv_lens_offsets):
                raise ValueError(
                    f"{type(self).__name__} keeps no device KV lengths to "
                    "apply kv_lens_offsets to")
            core.apply_batch_inputs(inputs)
            if context_requests is None:
                context_requests = inputs.context_requests
                prefill_cuda_graphs = inputs.prefill_cuda_graphs
        core.set_context_requests(context_requests, prefill_cuda_graphs)
        if not self.supports_explicit_offsets:
            self._check_offsets_match_lengths()
        self._prepare_core()
        for extension in self._attached_extensions():
            if extension is not core:
                extension.prepare()
        # Core corrections extensions made in their prepare (Rocket, Helix)
        # are part of the step's derivation.
        core.clear_lengths_changed()
        if inputs is not None and inputs.cross is not None:
            from .features import CrossAttentionMetadata
            self.ext(CrossAttentionMetadata).prepare_sub_metadata(inputs.cross)

    def _check_offsets_match_lengths(self) -> None:
        """Reject explicit boundaries this metadata's kernels would ignore."""
        for offsets, lengths, name in ((self.cu_q_seqlens, self.seq_lens,
                                        "cu_q_seqlens"),
                                       (self.cu_kv_seqlens, self.seq_lens_kv,
                                        "cu_kv_seqlens")):
            if offsets is None or lengths is None:
                continue
            expected = torch.zeros(lengths.numel() + 1, dtype=torch.int64)
            torch.cumsum(lengths.to(torch.int64), 0, out=expected[1:])
            if not torch.equal(offsets.cpu().to(torch.int64)[:expected.numel()],
                               expected):
                raise ValueError(
                    f"{type(self).__name__} derives sequence boundaries from "
                    f"seq_lens and cannot honor a {name} that differs from "
                    "their cumulative sum")

    def _prepare_core(self) -> None:
        """Derive the core's per-step state; ``prepare`` runs the extensions around it."""
        self.core().derive()

    def prepare_encoder_only(self,
                             inputs: Optional[BatchInputs] = None) -> None:
        """Encoder-only (no-KV-cache) forward setup.

        ``inputs`` replaces the batch description; without it the fields
        already set are used. The core defaults to the full ``prepare()``; a
        family with a leaner encoder-only path overrides it in its core.
        """
        if inputs is not None:
            self.core().apply_batch_inputs(inputs)
        self.core().prepare_encoder_only()
        self.core().clear_lengths_changed()

    def create_cuda_graph_metadata(self,
                                   max_batch_size: int,
                                   sub_cross_metadata: bool = False,
                                   max_draft_tokens: int = 0,
                                   buffers=None,
                                   encode_only: bool = False) -> Self:
        """
        Creates metadata for CUDA graph execution.
        CUDA graphs require to use pre-allocated buffers for all tensors in fields.
        Please do not re-allocate any tensors stored inside AttentionMetadata
        after the initial warmup run when you're using CUDA graphs.

        When encode_only is True, initialize seq_lens as ones(max_batch_size)
        and leave num_contexts at the caller's discretion.
        """
        if self.is_cuda_graph:
            return self

        cuda_graph_metadata = self.constructed_copy(
            is_cuda_graph=True,
            cuda_graph_buffers=buffers,
            **self.core().cuda_graph_copy_changes(max_batch_size,
                                                  max_draft_tokens))
        graph_core = cuda_graph_metadata.core()
        from .features import CrossAttentionMetadata
        cross = cuda_graph_metadata.find_ext(CrossAttentionMetadata)
        if cross is not None:
            cross.copy_sub_metadata_for_cuda_graph(max_batch_size,
                                                   max_draft_tokens, buffers,
                                                   encode_only)
        graph_core.reset_cuda_graph_lengths(
            max_batch_size,
            max_draft_tokens=max_draft_tokens,
            sub_cross_metadata=sub_cross_metadata,
            encode_only=encode_only)
        return cuda_graph_metadata

    def update_spec_dec_param(
            self,
            batch_size: int,
            spec_metadata: Optional['SpecMetadata'] = None,
            spec_tree_manager: Optional['SpecTreeManager'] = None,
            num_contexts: int = 0) -> None:
        """Write one step's draft-token layout; the settings are creation config.

        ``num_contexts`` is the number of context (prefill) requests in the
        mixed batch, occupying the leading rows of slot-storage buffers.
        Backends that consume gen-only slots (e.g. dynamic tree) must skip
        these rows to align with the XQA kernel's expected row layout.
        Metadata without speculative-decoding state ignores it.
        """
        update = self.find("update")
        if update is not None:
            update(batch_size, spec_metadata, spec_tree_manager, num_contexts)

    def create_cross_metadata(
        self,
        encoder_seq_lens: List[int],
        cross_kv_cache_manager: Union[KVCacheManager, KVCacheManagerV2,
                                      None] = None,
        encoder_num_cached_tokens_per_seq: Optional[List[int]] = None,
    ) -> "AttentionMetadata":
        """Build a cross-attention sub-metadata; see ``CrossAttentionMetadata``.

        Attaches ``CrossAttentionMetadata`` on first use, so only models with
        cross attention carry it.
        """
        from .features import CrossAttentionMetadata
        return self.add_ext(CrossAttentionMetadata).create_sub_metadata(
            encoder_seq_lens, cross_kv_cache_manager,
            encoder_num_cached_tokens_per_seq)

    @staticmethod
    def get_empty(buffers: Buffers,
                  tensor_shape: list[int],
                  dtype: torch.dtype,
                  cache_name: str,
                  capture_graph: bool = False) -> torch.Tensor:
        """
        Finds a compatible, reusable buffer from a cache or creates a new one.

        This function searches for a pre-allocated tensor (buffer) that can be
        reused for an operation involving a tensor with the shape of `tensor_shape`.

        The compatibility rules are: The buffer's total elements must be >= tensor_shape's.

        If a compatible buffer is found, it's returned immediately. Otherwise, a new
        buffer is allocated on the 'cuda' device with the give properties of 'tensor_shape' and 'dtype'.

        Args:
            tensor_shape: The required shape.
            dtype: The required dtype.
            cache_name: The key for the specific list of buffers to search in.
        Returns:
            An existing compatible buffer or a newly created one.
        """
        if buffers is None:
            return torch.zeros(tensor_shape, device='cuda', dtype=dtype)

        return buffers.get_buffer(tensor_shape, dtype, cache_name,
                                  capture_graph)

    @staticmethod
    def get_empty_like(buffers,
                       like_tensor: torch.Tensor,
                       cache_name: str,
                       capture_graph: bool = False) -> torch.Tensor:
        return AttentionMetadata.get_empty(
            buffers,
            like_tensor.shape,
            dtype=like_tensor.dtype,
            cache_name=cache_name,
            capture_graph=capture_graph,
        )


class PositionalEmbedder(Protocol):
    """
    A callable that can apply positional embedding
    """

    def __call__(self, position_ids: torch.Tensor, q: torch.Tensor,
                 k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


@dataclass(kw_only=True, unsafe_hash=True)
class RopeParams:
    dim: int = 0
    theta: float = 10000.0
    alpha: float = 1.0
    scale_type: RotaryScalingType = RotaryScalingType.none
    scale: float = 1.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    short_m_scale: float = 1.0
    long_m_scale: float = 1.0
    max_positions: int = 1024
    original_max_positions: int = 1024
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    mscale_all_dim: float = 0.0
    short_factor: Optional[Tuple[float]] = None
    long_factor: Optional[Tuple[float]] = None
    max_seq_len: Optional[int] = None
    duplicate_data: bool = False

    @staticmethod
    def from_config(config) -> "RopeParams":
        rope_params = RopeParams()

        hf_rope_parameters = getattr(config, 'rope_parameters', None)
        normalized_rope_parameters = hf_rope_parameters
        if hf_rope_parameters is not None:
            if set(hf_rope_parameters.keys()).issubset(
                    ALLOWED_ATTENTION_LAYER_TYPES):
                # Per-layer-type RoPE config (e.g. Gemma3 in transformers 5.x).
                # Pick "full_attention" as the default; callers override theta
                # for sliding-window layers independently.
                if "full_attention" in hf_rope_parameters:
                    normalized_rope_parameters = hf_rope_parameters[
                        "full_attention"]
                else:
                    fallback_key = next(iter(hf_rope_parameters))
                    logger.warning(
                        f"Per-layer-type rope_parameters has no 'full_attention' entry; "
                        f"falling back to '{fallback_key}'. Available layer types: "
                        f"{list(hf_rope_parameters.keys())}.")
                    normalized_rope_parameters = hf_rope_parameters[
                        fallback_key]
            config.update(normalized_rope_parameters)

        # get rotary parameters.
        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, 'head_dim', None)
        if not isinstance(head_dim, int):
            head_dim = hidden_size // num_attention_heads
        rope_scaling = getattr(config, 'rope_scaling', None)
        if rope_scaling is None and normalized_rope_parameters is not None:
            rope_scaling = normalized_rope_parameters
        rope_params.max_positions = config.max_position_embeddings
        rope_params.theta = get_hf_rope_theta(config, 10000.0)
        rope_percentage = (getattr(config, 'rotary_pct', None)
                           or getattr(config, 'partial_rotary_factor', None)
                           or 1.0)
        # rotary embedding dim.
        qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', None)
        rope_params.dim = (getattr(config, 'rotary_dim', None)
                           or getattr(config, 'rotary_emb_base', None)
                           or qk_rope_head_dim
                           or int(head_dim * rope_percentage))
        if qk_rope_head_dim is not None:
            rope_params.duplicate_data = True
        # rotary scaling.
        rope_params.scale_type = RotaryScalingType.none
        rope_params.scale = 1.0
        if rope_scaling is not None:
            rope_params.alpha = rope_scaling.get("alpha", 1.0)
            rotary_scaling_type = rope_scaling.get(
                "type", None) or rope_scaling.get("rope_type")
            rope_params.scale_type = RotaryScalingType.from_string(
                rotary_scaling_type)
            rope_params.scale = rope_scaling.get("factor", 1.0)
            rope_params.low_freq_factor = rope_scaling.get(
                "low_freq_factor", 1.0)
            rope_params.high_freq_factor = rope_scaling.get(
                "high_freq_factor", 4.0)
            rope_params.original_max_positions = getattr(
                config,
                "original_max_position_embeddings", None) or rope_scaling.get(
                    "original_max_position_embeddings", None) or 1024
            rope_params.beta_fast = rope_scaling.get("beta_fast", 32)
            rope_params.beta_slow = rope_scaling.get("beta_slow", 1)
            rope_params.mscale = rope_scaling.get("mscale", 1.0)
            rope_params.mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)
            if "short_factor" in rope_scaling:
                rope_params.short_factor = tuple(rope_scaling["short_factor"])
            if "long_factor" in rope_scaling:
                rope_params.long_factor = tuple(rope_scaling["long_factor"])
        # Workaround for DeepSeek V3 Lite since its rope_scaling is null in
        # config.json.
        elif config.model_type == "deepseek_v3":
            rope_params.scale_type = RotaryScalingType.yarn
        # Other metdadata for RoPE.
        rope_params.max_seq_len = getattr(config, 'max_seq_len', None)

        return rope_params

    def create_rope_const_params(self, interleave: bool = True):
        if self.dim == 0:
            return None, None

        RopeConstParams = namedtuple("RopeConstParams", ["inv_freq", "cos_sin"])
        extra_attrs = get_model_extra_attrs()
        if extra_attrs is not None:
            cache = extra_attrs.setdefault("rope_const_params", {})
            rope_const_params = cache.get((self, interleave), None)
            if rope_const_params is not None and rope_const_params.cos_sin(
            ) is not None:
                return (
                    rope_const_params.inv_freq()
                    if rope_const_params.inv_freq is not None else None,
                    rope_const_params.cos_sin(),
                )

        if self.scale_type == RotaryScalingType.yarn:
            rope_inv_freq, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_yarn(
                self.max_positions,
                self.dim,
                self.theta,
                self.scale,
                self.original_max_positions,
                self.beta_fast,
                self.beta_slow,
                self.mscale,
                self.mscale_all_dim,
                self.duplicate_data,
            )
        elif self.scale_type == RotaryScalingType.longrope:
            rope_inv_freq, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_long_rope(
                num_pos=self.max_positions,
                dim=self.dim,
                theta=self.theta,
                original_max_pos=self.original_max_positions,
                short_factor=self.short_factor,
                long_factor=self.long_factor,
                max_seq_len=self.max_seq_len,
                duplicate_data=self.duplicate_data,
            )
        else:
            rope_inv_freq, rope_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
                self.max_positions,
                self.dim,
                self.theta,
                self.scale,
                self.scale_type,
                rope_scaling_config={
                    "factor": self.scale,
                    "alpha": self.alpha,
                    "low_freq_factor": self.low_freq_factor,
                    "high_freq_factor": self.high_freq_factor,
                    "original_max_position_embeddings":
                    self.original_max_positions,
                },
                duplicate_data=self.duplicate_data,
            )
        if rope_inv_freq is not None:
            rope_inv_freq = torch.tensor(
                rope_inv_freq,
                dtype=torch.float32,
                device='cuda',
            )
        if not interleave:
            rope_cos_sin = rope_cos_sin.reshape(
                self.max_positions, -1,
                2)[:, :self.dim // 2, :].transpose(0, 2, 1).reshape(1, -1)
        rope_cos_sin = torch.tensor(
            rope_cos_sin,
            dtype=torch.float32,
            device='cuda',
        )
        if extra_attrs is not None:
            cache[(self, interleave)] = RopeConstParams(
                weakref.ref(rope_inv_freq)
                if rope_inv_freq is not None else None,
                weakref.ref(rope_cos_sin),
            )
        # One-shot log on cache miss (typically 2-4 times per model load).
        log_tensor_size("rope/new_table",
                        rope_cos_sin,
                        max_pos=self.max_positions,
                        dim=self.dim,
                        theta=self.theta,
                        scale_type=self.scale_type,
                        interleave=interleave)
        return rope_inv_freq, rope_cos_sin


@dataclass(kw_only=True, frozen=True)
class PositionalEmbeddingParams:
    type: PositionEmbeddingType
    embedder: Optional[PositionalEmbedder] = None

    # RoPE params
    rope: Optional[RopeParams] = None
    is_neox: bool = True

    # mRoPE params
    mrope_section: Optional[List[int]] = None
    mrope_interleaved: bool = False

    def __post_init__(self) -> None:
        if self.type.is_deferred():
            assert self.embedder is not None, f"{self.type} requires a not-none external embedder"
        else:
            assert self.embedder is None, f"Embedder must be None for {self.type}"

        if self.type.is_rope():
            assert self.rope is not None, f"{self.type} requires a not-none rope"


TMetadata = TypeVar("TMetadata", bound=AttentionMetadata)


class PredefinedAttentionMask(str, Enum):
    """
    Predefined attention mask types

    Attributes:
        CAUSAL: Use causal mask.
        FULL: do not use any mask
    """
    CAUSAL = "causal"
    FULL = "full"


class CustomAttentionMask(str, Enum):
    """
    Custom attention mask types
    """
    CUSTOM = "custom"


AttentionMask = Union[PredefinedAttentionMask, CustomAttentionMask]


@dataclass(kw_only=True, slots=True)
class AttentionForwardArgs:
    """Per-forward optional arguments for attention backends."""

    # Caller-facing output buffer, allocated here when absent. Native kernels
    # consume the per-phase slice in ``FmhaParams.output``, not this full buffer.
    output: Optional[torch.Tensor] = None
    output_sf: Optional[torch.Tensor] = None
    variable_window_token_starts: Optional[torch.Tensor] = cpp_field(
        dtype=torch.int32)
    variable_window_token_ends: Optional[torch.Tensor] = cpp_field(
        dtype=torch.int32)
    kv_scale_orig_quant: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    kv_scale_quant_orig: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    out_scale: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    out_scale_sf: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    latent_cache: Optional[torch.Tensor] = None
    q_pe: Optional[torch.Tensor] = None
    mrope_rotary_cos_sin: Optional[torch.Tensor] = None
    mrope_position_deltas: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    softmax_stats_tensor: Optional[torch.Tensor] = None
    attention_sinks: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    cu_q_seqlens: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    cu_kv_seqlens: Optional[torch.Tensor] = cpp_field(dtype=torch.int32)
    fmha_scheduler_counter: Optional[torch.Tensor] = cpp_field(
        dtype=torch.uint32)
    mla_bmm1_scale: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    mla_bmm2_scale: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    quant_q_buffer: Optional[torch.Tensor] = None
    cross_kv: Optional[torch.Tensor] = None
    relative_attention_bias: Optional[torch.Tensor] = None
    quant_scale_qkv: Optional[torch.Tensor] = cpp_field(dtype=torch.float32)
    dsv4_inv_rope_cos_sin_cache: Optional[torch.Tensor] = cpp_field(
        dtype=torch.float32)
    enable_dsv4_epilogue_fusion: bool = False

    # Fused kv_a_layernorm, DSv4 sparse context path. When set, `latent_cache` is the
    # RAW kv_a_proj output and the context RoPE kernel norms it before RoPE + quant +
    # paged write, so the caller drops its own RMSNorm and concat.
    kv_norm_weight: Optional[torch.Tensor] = None
    kv_norm_eps: float = 1e-6

    attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL
    attention_input_type: AttentionInputType = AttentionInputType.mixed
    attention_window_size: Optional[int] = None
    attention_mask_data: Optional[torch.Tensor] = cpp_field(dtype=torch.bool)
    relative_attention_max_distance: int = 0
    chunked_prefill_buffer_batch_size: int = 1
    skip_mla_rope_generation: bool = False
    sage_attn_num_elts_per_blk_q: int = 0
    sage_attn_num_elts_per_blk_k: int = 0
    sage_attn_num_elts_per_blk_v: int = 0
    sage_attn_qk_int8: bool = False
    sage_attn_smooth_k: bool = False
    # Subtract the per-channel V mean before quantizing V. Applied by FMHA libraries that add
    # the mean back to the output; the others quantize V unsmoothed.
    sage_attn_smooth_v: bool = False
    # SageAttention inputs that the attention core quantized for the FMHA libraries
    # taking quantized inputs; the attention op quantizes inside its kernels.
    sage_quantized: Optional["SageQuantized"] = None
    # Per-tensor FP8 recipe of the PrimTS FMHA library. Quantize Q/K and/or V to
    # E4M3 with one scale each, folded into the softmax and output scales.
    qk_fp8_per_tensor: bool = False
    v_fp8_per_tensor: bool = False
    # VC-Attention recipe of the PrimTS FMHA library. The caller prepares the
    # operands (K in grouped token order, V as E4M3 tile residuals with its
    # scale and means, under QK8 the rotated E4M3 Q and K with their block
    # scales) and the library runs the matching plan on them.
    vc_attention: bool = False
    vc_operands: Optional["VCAttentionOperands"] = None

    # Packed QKV for non-MLA attention. MLA always passes a separate query.
    is_fused_qkv: bool = False
    update_kv_cache: bool = True
    # Attention output gate [num_tokens, ...] offered to the FMHA library: one that multiplies sigmoid(gate) into
    # the output sets output_gate_applied, and the caller applies the gate otherwise.
    output_gate: Optional[torch.Tensor] = None
    output_gate_applied: bool = False
    # Paged FP8 query / K / V the caller already wrote for this generation-only call (fmha/gqa_decode.py).
    kv_append: Optional["KvAppendPlan"] = None
    # Optional normalized diffusion timestep for timestep-varying sparse attention.
    timestep: Optional[Union[torch.Tensor, float]] = None
    # Dense-prefix (0) or sparse (1) phase of a timestep-scheduled sparse
    # algorithm, resolved on the host by the caller. When set, backends use it
    # instead of reading ``timestep``, which CUDA Graph capture cannot do.
    sparse_attn_phase: Optional[int] = None

    sparse_backend_args: Optional[SparseBackendForwardArgs] = None
    sparse_runtime_params: SparseRuntimeParams = field(
        default_factory=SparseRuntimeParams)

    @property
    def mask_type(self) -> int:
        """Return the mask type this forward pass asks the native attention op for."""
        if self.attention_mask == PredefinedAttentionMask.CAUSAL:
            return int(AttentionMaskType.causal)
        if self.attention_mask == PredefinedAttentionMask.FULL:
            return int(AttentionMaskType.padding)
        raise ValueError(
            f"Unexpected attention mask type: {self.attention_mask!r}")

    @property
    def sage_attn_qk_dtype(self) -> Optional[torch.dtype]:
        """Quantized Q/K dtype of the SageAttention recipe, or ``None`` without one."""
        if (self.sage_attn_num_elts_per_blk_q <= 0
                and self.sage_attn_num_elts_per_blk_k <= 0
                and self.sage_attn_num_elts_per_blk_v <= 0):
            return None
        return torch.int8 if self.sage_attn_qk_int8 else torch.float8_e4m3fn


_ATTENTION_FORWARD_ARGS_FIELDS = frozenset(
    AttentionForwardArgs.__dataclass_fields__)


def merge_attention_forward_args(
        forward_args: Optional[AttentionForwardArgs],
        kwargs: Dict[str, Any]) -> AttentionForwardArgs:
    """Merge legacy attention kwargs into explicit forward arguments."""
    unknown_kwargs = sorted(set(kwargs) - _ATTENTION_FORWARD_ARGS_FIELDS)
    if unknown_kwargs:
        raise ValueError(
            f"Unknown attention forward arguments: {unknown_kwargs}")
    if forward_args is not None:
        if kwargs:
            raise ValueError(
                "Pass attention forward options either through forward_args "
                f"or as legacy kwargs, not both: {sorted(kwargs)}")
        return forward_args
    return AttentionForwardArgs(**kwargs)


class AttentionBackend(Generic[TMetadata]):
    """
    Base class for attention backends.
    """
    Metadata: Type[TMetadata] = AttentionMetadata
    # Extensions every metadata of this backend carries, e.g. a sparse
    # algorithm's state; see extensions.py.
    metadata_extensions: Tuple[Type[MetadataExtension], ...] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        extensions = cls.__dict__.get("metadata_extensions")
        carried = cls.Metadata.backend_extension_types
        if extensions and not all(
                any(issubclass(have, wanted) for have in carried)
                for wanted in extensions):
            # Metadata built through ``cls.Metadata`` carries the backend's
            # extensions, as a subclass of the inherited metadata class.
            base = cls.Metadata
            cls.Metadata = type(
                f"{cls.__name__}Metadata", (base, ), {
                    "__module__":
                    cls.__module__,
                    "backend_extension_types":
                    (*base.backend_extension_types, *extensions),
                })

    @classmethod
    def create_metadata(cls, **kwargs) -> TMetadata:
        """Build this backend's metadata with its extensions attached."""
        return cls.Metadata(**kwargs)

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """
        Initialize the backend.
        Args:
            layer_idx (int): The index of the attention layer in the model.
            num_heads (int): The number of query heads.
            head_dim (int): The size of each attention head (hidden_size // num_heads).
            num_kv_heads (int): The number of kv heads. Defaults to num_heads if None.
            quant_config (QuantConfig): Optional quantization configuration. If None, no quantization is applied.
        """
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads or self.num_heads
        self.quant_config = quant_config

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        """
        To support mixed quantization mode, self.quant_config can be modified after __init__ is called.
        Any states or set up related to self.quant_config must be moved to this function, which is called
        after self.quant_config is reset.
        """
        self.quant_config = new_quant_config

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: TMetadata,
                forward_args: Optional[AttentionForwardArgs] = None,
                **kwargs) -> torch.Tensor:
        """
        Update KV Cache and perform the attention operation.
        Args:
            q (torch.Tensor): Query tensor with shape (num_q_tokens, num_heads * head_dim),
                              or QKV tensor with shape (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim).
            k (Optional[torch.Tensor]): Key tensor with shape (num_new_kv_tokens, num_kv_heads * head_dim),
                                        or KV tensor with shape (num_new_kv_tokens, (2 * num_kv_heads) * head_dim),
                                        or None: if QKV tensor is provided, or there's no new kv token.
            v (Optional[torch.Tensor]): Value tensor with shape (num_new_kv_tokens, num_kv_heads * head_dim),
                                        or None if QKV tensor is provided, or there's no new kv token.
            metadata (AttentionMetadata): Metadata for the attention operation.
            forward_args (AttentionForwardArgs): Per-forward optional attention arguments.
        Returns:
            torch.Tensor with shape (num_q_tokens, num_heads * head_dim)
        """
        raise NotImplementedError

    @classmethod
    def support_fused_rope(cls) -> bool:
        return False

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_mla(cls) -> bool:
        return False

    @classmethod
    def support_fp4_kv_cache(cls) -> bool:
        """Whether the backend can execute attention with an FP4 KV cache."""
        return False

    @classmethod
    def support_multi_item_scoring(cls) -> bool:
        return False

    @classmethod
    def runtime_workspace_bytes_per_token(cls, model_config: "ModelConfig",
                                          mapping: Mapping) -> int:
        """Per-token bytes to reserve for a workspace this backend stages whose size scales with a
        runtime quantity the KV-cache estimator does not drive to its serving maximum while profiling
        (e.g. ``total_kv_len``, inflated by KV-cache reuse). Default ``0`` -- correct for every backend
        that does not stage such context-MLA data.

        A non-zero rate is reserved from the KV budget by the estimator, and the scheduler caps the
        driving sum at what that reserve covers, so the declared buffer stays within its reservation
        (buffers the backend does not declare are still unaccounted for). Keep the rate identical to the
        runtime allocation's per-token cost or use a documented conservative upper bound. See
        ``ATTENTION_DEVELOPER_GUIDE.md`` §2.3.
        """
        return 0

    @classmethod
    def runtime_workspace_is_chunked_prefill_bounded(
            cls, model_config: "ModelConfig") -> bool:
        """Whether chunked prefill bounds the declared workspace by the current chunk.

        Backends that still stage data proportional to the complete attended
        prefix must return ``False`` so KV-cache sizing reserves headroom for
        that prefix.
        """
        return True

    @classmethod
    def runtime_workspace_fixed_bytes(
        cls,
        model_config: "ModelConfig",
        mapping: Mapping,
        *,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        enable_cuda_graph: bool,
    ) -> int:
        """Capacity-sized workspace not bounded by summed attended KV length.

        The estimator reserves the full capacity alongside the profiled peak,
        covering arenas first allocated after the peak prefill step.
        """
        return 0

    def create_output(self, q: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        Create the output tensors for the attention operation.
        """
        num_tokens = q.shape[0]
        hidden_size = self.num_heads * self.head_dim
        return [q.new_empty([num_tokens, hidden_size], dtype=q.dtype)]


@dataclass(kw_only=True, unsafe_hash=True)
class MLAParams:
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    qk_nope_head_dim: int = 0
    v_head_dim: int = 0
    rope_append: bool = True
    predicted_tokens_per_seq: int = 1
    chunked_prefill_buffer_batch_size: int = 1
    hidden_size: int = 0
