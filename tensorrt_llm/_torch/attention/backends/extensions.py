# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Feature state attached to attention metadata.

An ``AttentionMetadata`` holds the batch layout every attention backend needs:
request counts, sequence and KV lengths, the paged KV cache and the workspace.
Everything a feature, a sparse algorithm or a model adds on top lives in a
``MetadataExtension`` attached to it. The metadata drives each extension
through the hooks below at fixed points of its lifecycle, and callers reach an
extension's state with ``metadata.ext(T)`` or ``metadata.find_ext(T)``.

Per-step order, all driven by the metadata:

1. ``prepare_scope()`` wraps the whole prepare.
2. ``pre_derive_lengths()`` runs before the core derives KV lengths.
3. ``override_kv_lens(kv_lens)`` may rewrite the host KV lengths the core
   just derived, before they are published to the device.
4. ``prepare()`` runs after the core, in dependency order.
5. ``post_derive()`` runs after every extension prepared.

Outside ``prepare()``: ``allocate()`` on creation and on every CUDA graph or
cross-attention copy, ``begin_forward()`` before the model forward,
``on_kv_lens_changed()`` after device-side length changes,
``on_spec_dec_param()`` after the speculative-decoding parameters change,
``invalidate_length_derived_buffers()`` whenever the core invalidates its own
length prefixes, and ``draft_swap()`` / ``draft_restore()`` around a
draft-model forward. ``token_major_gen_view()`` and ``mla_scheduler_buffers()``
let an extension replace the core's generation-row presentation.
"""

from __future__ import annotations

import contextlib
import copy
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ContextManager,
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

if TYPE_CHECKING:
    import torch

    from ...memory_buffer_utils import Buffers
    from ...pyexecutor.llm_request import LlmRequest
    from ...speculative.spec_tree_manager import SpecTreeManager
    from .interface import AttentionMetadata
    from .sparse.dsa.metadata import TokenMajorGenView

E = TypeVar("E", bound="MetadataExtension")


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

    def __init__(self, metadata: "AttentionMetadata") -> None:
        self.metadata = metadata

    def clone_for(self, metadata: "AttentionMetadata") -> "MetadataExtension":
        """Copy this extension for a CUDA graph or cross-attention copy.

        The default shallow copy shares every attribute with this extension;
        ``allocate()`` then runs on the copy, so graph-stable buffers allocated
        there are rebound per copy. Override to give a copy fresh per-copy
        state that ``allocate()`` does not rebuild.
        """
        clone = copy.copy(self)
        clone.metadata = metadata
        return clone

    def allocate(self, buffers: Optional["Buffers"]) -> None:
        """Allocate persistent buffers; runs again on every metadata copy."""

    def prepare_scope(self) -> ContextManager[None]:
        """A context the whole ``prepare()`` runs in."""
        return contextlib.nullcontext()

    def pre_derive_lengths(self) -> None:
        """Adjust the per-request inputs before the core derives KV lengths."""

    def override_kv_lens(self, kv_lens: "torch.Tensor") -> None:
        """Rewrite the host KV lengths in place before they are published.

        ``kv_lens`` holds the per-sequence KV lengths, without the extra KV
        tokens, for the whole batch.
        """

    def prepare(self) -> None:
        """Prepare this extension's per-step state after the core."""

    def post_derive(self) -> None:
        """Adjust the core's published views after every extension prepared."""

    def begin_forward(self) -> None:
        """Reset per-forward scratch state before the model forward."""

    def on_kv_lens_changed(self, spec_dec_step: bool) -> None:
        """React to device-side KV-length changes.

        ``spec_dec_step`` is True when a speculative-decoding sub-step changed
        the lengths (``update_for_spec_dec``) and False for an overlap or
        draft-correction update (``on_update_kv_lens``).
        """

    def on_spec_dec_param(
        self,
        max_draft_len: int,
        max_total_draft_tokens: int,
        spec_tree_manager: Optional["SpecTreeManager"],
    ) -> None:
        """React to new speculative-decoding parameters.

        Runs after the core stored its own spec-dec fields, with the arguments
        of ``update_spec_dec_param`` the core does not keep.
        """

    def invalidate_length_derived_buffers(self) -> None:
        """Mark buffers derived from this step's lengths stale.

        Called whenever the core invalidates its own layer-invariant length
        prefixes: at the start of every prepare and on each KV-length change.
        """

    def token_major_gen_view(self) -> Optional["TokenMajorGenView"]:
        """A one-row-per-query-token generation presentation, or None.

        The core presents the first non-None view to the MLA RoPE and
        sparse-MLA generation ops.
        """
        return None

    def mla_scheduler_buffers(
        self, num_heads: int
    ) -> Optional[Tuple["torch.Tensor", "torch.Tensor"]]:
        """Cumulative MLA generation Q rows / KV lengths replacing the core's, or None."""
        return None

    def on_spec_dec_restored(self) -> None:
        """Rebind views after ``restore_from_spec_dec`` restored the core tensors."""

    def needs_full_block_table_width(self) -> bool:
        """Whether every block-table column must be staged each step.

        True when device-side updates may advance KV lengths past the host
        snapshot, so the core cannot cap the staged width at the batch maximum.
        """
        return False

    def draft_swap(self) -> Any:
        """Point this extension at the draft model's state; return what to restore."""
        return None

    def draft_restore(self, saved: Any) -> None:
        """Undo ``draft_swap``."""

    def swap_attrs_for_draft(self, draft_values: Dict[str, Any]) -> Dict[str, Any]:
        """Rebind attributes to their draft values; return the originals.

        The originals are also kept in ``draft_replay_swapped_attrs`` until
        ``restore_swapped_attrs``, so the strict CUDA-graph buffer check
        validates the target tensors while a draft forward is bound.
        """
        swapped = self.__dict__.setdefault("draft_replay_swapped_attrs", {})
        saved = {}
        for name, value in draft_values.items():
            saved[name] = getattr(self, name, None)
            swapped.setdefault(name, saved[name])
            setattr(self, name, value)
        return saved

    def restore_swapped_attrs(self, saved: Dict[str, Any]) -> None:
        """Undo ``swap_attrs_for_draft``."""
        for name, value in saved.items():
            setattr(self, name, value)
        self.__dict__["draft_replay_swapped_attrs"] = {}

    def adp_token_counts(self) -> Optional[List[int]]:
        """Local counts of the extra stages this extension adds to the attention-DP exchange."""
        return None

    def set_adp_token_counts(self, counts: List[List[int]], input_counts: List[int]) -> None:
        """Receive every rank's counts for the stages ``adp_token_counts`` added.

        ``input_counts`` holds every rank's input-stage count.
        """

    def workspace_bytes(self) -> int:
        """Extra attention workspace this extension needs."""
        return 0

    def kv_cache_pools(self) -> Optional[Tuple["torch.Tensor", "torch.Tensor", int]]:
        """The KV pools the attention kernels read instead of the manager's, or None.

        Returns host pool pointers, host pool mapping and tokens per block.
        """
        return None

    def on_runtime_views_bound(self) -> None:
        """React to the core rebinding its per-forward ``*_runtime`` views."""

    def bind_context_requests(
        self, requests: List["LlmRequest"], *, prefill_cuda_graphs: bool
    ) -> None:
        """Receive the scheduled context requests before ``prepare()``.

        ``prefill_cuda_graphs`` tells whether context forwards may run in CUDA
        graphs.
        """

    def defer_decode_outputs(self) -> Optional[ContextManager[None]]:
        """A scope for preparing a decode CUDA graph's metadata, or None.

        Inside it, outputs that the graph's own preprocessing derives after
        ``on_update_kv_lens`` may be left unprepared.
        """
        return None


def resolve_extension_order(
    types: Iterable[Type[MetadataExtension]],
) -> List[Type[MetadataExtension]]:
    """Order extension types so every requirement precedes its dependents.

    Requirements that are not listed are added. Listing order is otherwise
    preserved.
    """
    ordered: List[Type[MetadataExtension]] = []
    visiting: set = set()

    def visit(cls: Type[MetadataExtension]) -> None:
        if cls in ordered:
            return
        if cls in visiting:
            raise ValueError(f"Metadata extension requirements form a cycle at {cls.__name__}.")
        visiting.add(cls)
        for required in cls.requires:
            visit(required)
        visiting.discard(cls)
        ordered.append(cls)

    for cls in types:
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

    def find(self, cls: Type[E]) -> Optional[E]:
        found = self._by_type.get(cls)
        if found is not None:
            return found
        for extension in self._by_type.values():
            if isinstance(extension, cls):
                return extension
        return None

    def attach(
        self, metadata: "AttentionMetadata", types: Sequence[Type[MetadataExtension]]
    ) -> List[MetadataExtension]:
        """Attach the given types and their requirements; return the new ones."""
        added = []
        for cls in resolve_extension_order(types):
            if cls not in self._by_type:
                extension = cls(metadata)
                self._by_type[cls] = extension
                added.append(extension)
        return added

    def replace(self, extension: MetadataExtension) -> None:
        """Put an already-built extension in place of any of its type."""
        self._by_type[type(extension)] = extension

    def remove(self, cls: Type[MetadataExtension]) -> None:
        """Detach the extension of type ``cls``, if any."""
        found = self.find(cls)
        if found is not None:
            del self._by_type[type(found)]

    def bound_to(self, metadata: "AttentionMetadata") -> "ExtensionRegistry":
        """This registry, owned by ``metadata``.

        Returns ``self`` when ``metadata`` already owns it; otherwise a new
        registry holding a ``clone_for`` copy of each extension, so a shallow
        copy of the metadata never shares the registry or its extensions with
        its source.
        """
        if self.owner is metadata:
            return self
        clone = ExtensionRegistry(metadata)
        clone._by_type = {cls: e.clone_for(metadata) for cls, e in self._by_type.items()}
        return clone
