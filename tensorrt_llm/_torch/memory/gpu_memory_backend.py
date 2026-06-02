# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""GPU Memory Backend protocol and GMS implementation.

Defines a thin protocol that adapts the upstream GPU Memory Service
(``gpu_memory_service`` from ``ai-dynamo/dynamo``) into TRT-LLM's
``ModelLoader`` pipeline. The protocol intentionally exposes only the
operations TRT-LLM needs at specific points in the loading lifecycle —
all heavy lifting (CUDA VMM, FD passing, zero-copy tensor construction)
is delegated to the GMS library's stable public Python primitives.

Design notes:
- We deliberately avoid the upstream ``setup_gms()`` monkey-patch entry
  point. TRT-LLM owns the integration policy in ``ModelLoader``; this
  backend is a thin adapter over the GMS library's primitive API.
- The protocol has a single point of contact per concern, so when the
  upstream GMS Python API drifts, only the methods on ``GMSBackend``
  need to change — the call sites in ``model_loader.py`` are stable.

Operating modes:
- **RW (Read-Write)**: First worker loads weights via the normal
  checkpoint loader pipeline, with allocations captured by a GMS-managed
  CUDA memory pool. After loading, weights are committed for read-only
  access by other workers and the client transitions to RO mode in place.
- **RO (Read-Only)**: Subsequent workers zero-copy import already-committed
  weights from the GMS pool. ``post_load_weights()`` must run BEFORE
  materialization so that module aliases are set up correctly.
"""

from contextlib import contextmanager
from typing import Iterator, Optional, Protocol, runtime_checkable

import torch
from torch import nn

from tensorrt_llm._torch.weight_sharing import SourceIdentity
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


@runtime_checkable
class GPUMemoryBackend(Protocol):
    """Thin abstraction over GPU memory services for API stability."""

    def connect(self) -> bool:
        """Establish a session with the backend. Returns True on success."""
        ...

    @property
    def is_rw(self) -> Optional[bool]:
        """Whether this backend was granted RW (vs RO). None pre-connect."""
        ...

    def has_committed_weights(self) -> bool:
        """Whether committed weights are ready to be imported (RO-ready)."""
        ...

    def mem_pool_scope(
        self,
        device: Optional[torch.device] = None,
    ) -> Iterator[None]:
        """Return a context manager scoping CUDA allocations to the backend pool."""
        ...

    def materialize_module(self, model: nn.Module) -> None:
        """Zero-copy import committed weights into model params (RO path)."""
        ...

    def get_source_identity(self) -> Optional["SourceIdentity"]:
        """Return the writer's committed SourceIdentity, if available (RO)."""
        ...

    def finalize_write(self, model: nn.Module) -> int:
        """Register, commit, and transition to RO. Returns bytes committed."""
        ...

    def move_untracked_params(self, model: nn.Module) -> None:
        """Move stray params not in the backend pool into shared memory."""
        ...

    def cleanup(self) -> None:
        """Release all resources and disconnect."""
        ...


# String mode -> upstream RequestedLockType. Resolved lazily inside connect()
# so importing this module does not require the GMS library to be installed.
_MODE_ALIASES = ("rw", "ro", "auto")


class GMSBackend:
    """Concrete ``GPUMemoryBackend`` using ``gpu_memory_service`` (GMS).

    GMS is a multi-process GPU memory manager (out-of-process server,
    in-process clients) that lets multiple inference instances share weight
    bytes via CUDA VMM mappings and Unix-socket FD passing.

    This adapter calls the GMS library's stable per-call primitives
    (``GMSClientMemoryManager`` + the helpers in
    ``gpu_memory_service.client.torch.allocator`` and ``.module``) and
    composes them into the methods TRT-LLM's ``ModelLoader`` invokes for
    the ``LoadFormat.GMS`` branch. We intentionally do **not** use the
    upstream ``setup_gms()`` monkey-patch — TRT-LLM owns the integration
    policy and we keep the dependency surface narrow and explicit.
    """

    DEFAULT_TAG = "weights"

    def __init__(
        self,
        socket_path: Optional[str],
        mapping: Mapping,
        mode: str = "auto",
        tag: str = DEFAULT_TAG,
    ) -> None:
        """Initialize the GMS backend.

        Args:
            socket_path: Unix domain socket path for the per-GPU GMS daemon.
                When ``None``, the default per-GPU UUID-keyed path from
                ``gpu_memory_service.common.utils.get_socket_path`` is used
                (resolved lazily inside :meth:`connect`).
            mapping: TRT-LLM distributed ``Mapping`` for TP/PP rank info.
            mode: Operating mode — ``"auto"`` (RW first, RO after committed),
                ``"rw"`` (require RW), or ``"ro"`` (require RO).
            tag: Logical tag identifying this weight set in the GMS server.
                Default ``"weights"`` matches the GMS library convention.
        """
        if mode not in _MODE_ALIASES:
            raise ValueError(f"GMS mode must be one of {_MODE_ALIASES}, got {mode!r}")

        self._socket_path = socket_path
        self._mapping = mapping
        self._mode = mode
        self._tag = tag
        self._device_index = torch.cuda.current_device()

        # Late-bound state, populated by connect()
        self._client = None  # GMSClientMemoryManager (lazy-typed)
        self._is_rw: Optional[bool] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to the per-GPU GMS daemon and acquire a session.

        Returns ``True`` on success. On import failure (library not
        installed) or socket failure, returns ``False`` and logs a warning
        — callers are expected to surface a useful error to the user.
        """
        try:
            from gpu_memory_service.client.torch.allocator import (
                get_or_create_gms_client_memory_manager,
            )
            from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
            from gpu_memory_service.common.utils import get_socket_path
            from gpu_memory_service.integrations.common.patches import patch_empty_cache
        except ImportError:
            logger.warning(
                "gpu_memory_service library not installed; cannot use "
                "LoadFormat.GMS. Install from "
                "https://github.com/ai-dynamo/dynamo/tree/main/lib/gpu_memory_service"
            )
            return False

        mode_map = {
            "rw": RequestedLockType.RW,
            "ro": RequestedLockType.RO,
            "auto": RequestedLockType.RW_OR_RO,
        }

        # Resolve default socket path against the upstream convention so
        # multiple TRT-LLM instances on the same node automatically share
        # the same per-GPU daemon.
        socket_path = self._socket_path
        if socket_path is None:
            socket_path = get_socket_path(self._device_index, self._tag)
        self._socket_path = socket_path

        try:
            self._client = get_or_create_gms_client_memory_manager(
                socket_path,
                self._device_index,
                mode=mode_map[self._mode],
                tag=self._tag,
            )
        except Exception as e:
            # TODO(GMS-API): narrow once upstream documents the expected
            # client-creation failure modes (socket missing, connection
            # refused, daemon crashed). Today the upstream API doesn't
            # enumerate them, so the broad catch protects the
            # "GMS optional / fall through to disk" guarantee at the cost
            # of also swallowing programming bugs (e.g. signature drift).
            logger.warning(
                "Failed to connect to GMS at %s (mode=%s, tag=%s): %s",
                socket_path,
                self._mode,
                self._tag,
                e,
            )
            self._client = None
            self._is_rw = None
            return False

        self._is_rw = self._client.granted_lock_type == GrantedLockType.RW

        # Process-wide safety patch: torch.cuda.empty_cache() can segfault
        # when it tries to free GMS-backed VMM regions. Upstream's
        # patch_empty_cache() rebinds torch.cuda.empty_cache to a GMS-aware
        # variant that skips those regions.
        #
        # Process-wide is intentional, not a leak: torch internals, MoE
        # balancer, draft-model setup, and unrelated TRT-LLM call sites all
        # invoke empty_cache and would crash otherwise. A per-call-site
        # wrapper would miss the calls we don't own.
        #
        # TODO(GMS-API): replace with a TRT-LLM-owned implementation once
        # upstream's API stabilizes (see TODO on move_untracked_params).
        # Idempotent and cheap.
        try:
            patch_empty_cache()
        except Exception as e:
            logger.debug("GMS patch_empty_cache failed (non-fatal): %s", e)

        logger.info(
            "Connected to GMS at %s (mode=%s, granted=%s, tag=%s)",
            socket_path,
            self._mode,
            "RW" if self._is_rw else "RO",
            self._tag,
        )
        return True

    @property
    def is_rw(self) -> Optional[bool]:
        """Whether this client holds the writer lock for the GMS pool.

        Returns:
            ``True`` if RW (writer) was granted, ``False`` if RO (reader)
            was granted, ``None`` before :meth:`connect` has been called
            successfully or after :meth:`cleanup` has run. Transitions
            from ``True`` to ``False`` after :meth:`finalize_write`.
        """
        return self._is_rw

    def has_committed_weights(self) -> bool:
        """Whether committed weights are available for RO materialization.

        Reports the granted lock type at the time of call, which only
        becomes ``RO`` once a writer in this or another process has
        published a finalized layout for ``tag``.

        Returns:
            ``True`` if the granted lock is ``RO`` (committed weights
            exist and can be zero-copy mapped); ``False`` otherwise,
            including pre-connect, post-cleanup, and any error paths.
            Never raises — best-effort by design so callers can use it
            as a precondition check before invoking
            :meth:`materialize_module`.
        """
        if self._client is None:
            return False
        try:
            from gpu_memory_service.common.locks import GrantedLockType

            return self._client.granted_lock_type == GrantedLockType.RO
        except Exception:
            return False

    # ------------------------------------------------------------------
    # RW path: scope allocations into the GMS pool, then finalize
    # ------------------------------------------------------------------

    @contextmanager
    def mem_pool_scope(
        self,
        device: Optional[torch.device] = None,
    ) -> Iterator[None]:
        """Context manager scoping CUDA allocations to the GMS pool.

        All ``torch.empty`` / ``torch.zeros`` / etc. allocations inside
        this scope are routed through the GMS pluggable allocator and
        land in the shared memory region for the configured ``tag``.
        Allocations made via paths that bypass the active allocator
        (e.g. C++ ops that call ``cudaMalloc`` directly) escape this
        scope and must be swept by :meth:`move_untracked_params`
        before :meth:`finalize_write` is called.

        Args:
            device: Target CUDA device. Defaults to
                ``torch.device('cuda', current_device())``.

        Yields:
            ``None`` — the value is unused; this is a scoping context.

        Raises:
            RuntimeError: If ``connect()`` has not been called yet, or
                if the granted lock is RO (only the writer may allocate
                into the pool).
        """
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")
        if self._is_rw is False:
            raise RuntimeError(
                "GMS mem_pool_scope() is only valid in RW mode (this client was granted RO)."
            )

        from gpu_memory_service.client.torch.allocator import gms_use_mem_pool

        target_device = device
        if target_device is None:
            target_device = torch.device("cuda", self._device_index)

        with gms_use_mem_pool(self._tag, target_device):
            yield

    def move_untracked_params(self, model: nn.Module) -> None:
        """Migrate parameters allocated outside the GMS pool into it.

        Some parts of TRT-LLM's loading pipeline (e.g. ``post_load_weights``,
        ``model.to("cuda")``) may allocate buffers outside the
        ``mem_pool_scope`` context. ``finalize_write`` requires every
        registered tensor to live in the GMS pool, so we copy any stray
        CUDA params into freshly created GMS mappings of the same size.

        Mirrors ``gpu_memory_service.integrations.trtllm.model_loader.
        _move_untracked_params`` so behavior matches the upstream
        reference integration.

        Args:
            model: The ``nn.Module`` to scan for stray CUDA parameters.
                Buffers (``tensor_type != 'parameter'``), CPU tensors,
                and tensors already backed by a GMS mapping are skipped.

        Raises:
            RuntimeError: If ``connect()`` has not been called yet.
        """
        # TODO(GMS-API): Clean up once the stable GMS API becomes available.
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")

        from gpu_memory_service.client.torch.module import _iter_module_tensors
        from gpu_memory_service.client.torch.tensor import _tensor_from_pointer

        gms_client = self._client
        device_index = self._device_index
        # Map original storage base ptr -> new GMS base_va. Serves two roles:
        #   1. Dedups create_mapping() / copy_() calls when multiple tensors
        #      share a storage (e.g. tied embeddings, sliced fused params).
        #   2. Lets us rebind every view of a shared storage to the SAME
        #      GMS mapping, so all tied/aliased parameters end up backed
        #      by the GMS pool. Without this, only the first encountered
        #      view was rebound; subsequent views still pointed at the
        #      original non-GMS storage and were missed by finalize_write.
        storage_to_gms_va: dict[int, int] = {}

        # No torch.no_grad() guard: ``tensor.data = X`` bypasses autograd by
        # definition (it's a data-attribute assignment, not an autograd-tracked
        # operation), and ``replacement.copy_(tensor)`` is an in-place op on a
        # freshly created tensor with no autograd history. Model loading also
        # runs outside any grad-enabled context.
        for _name, tensor, tensor_type in _iter_module_tensors(model):
            if tensor_type != "parameter" or tensor is None or not tensor.is_cuda:
                continue

            # GMS tracks whole storage allocations, so use the storage
            # base pointer instead of a tensor view's offset pointer.
            storage_base_ptr = tensor.untyped_storage().data_ptr()
            if _ptr_in_gms(gms_client, int(storage_base_ptr)):
                continue

            base_va = storage_to_gms_va.get(storage_base_ptr)
            first_for_this_storage = base_va is None
            if first_for_this_storage:
                nbytes = _storage_nbytes(tensor)
                base_va = gms_client.create_mapping(size=nbytes, tag=self._tag)
                storage_to_gms_va[storage_base_ptr] = base_va

            replacement = _tensor_from_pointer(
                int(base_va),
                list(tensor.shape),
                list(tensor.stride()),
                tensor.dtype,
                device_index,
            )
            # Copy original bytes exactly once per storage; subsequent
            # views of the same storage just rebind onto the same GMS
            # mapping without recopying.
            if first_for_this_storage:
                replacement.copy_(tensor)
            tensor.data = replacement

    def finalize_write(self, model: nn.Module) -> int:
        """Register tensors, commit them, and transition this client to RO.

        After this returns successfully, ``is_rw`` flips to ``False``
        and other instances on the same node connecting in ``"auto"``
        or ``"ro"`` mode will receive a zero-copy RO mapping of the
        committed layout. The transition is in-place: this client
        keeps the same socket connection.

        Mirrors ``gpu_memory_service.integrations.common.utils.
        finalize_gms_write``.

        Args:
            model: The fully-loaded ``nn.Module`` whose CUDA parameters
                will be registered with the GMS daemon. Every parameter
                must already live in the GMS pool — call
                :meth:`move_untracked_params` first to sweep strays
                allocated outside :meth:`mem_pool_scope`.

        Returns:
            Total bytes committed to the GMS pool for this ``tag``.

        Raises:
            RuntimeError: If ``connect()`` has not been called yet, or
                if the granted lock is RO (only the writer may commit).
        """
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")
        if self._is_rw is False:
            raise RuntimeError("GMS finalize_write() is only valid in RW mode.")

        from gpu_memory_service.integrations.common.utils import finalize_gms_write

        bytes_committed = int(finalize_gms_write(self._client, model))
        # finalize_gms_write internally commits and reconnects in RO mode,
        # so we mirror that state transition here.
        self._is_rw = False
        logger.info(
            "GMS RW->RO: committed %.2f GiB at %s (tag=%s)",
            bytes_committed / (1 << 30),
            self._socket_path,
            self._tag,
        )
        return bytes_committed

    # ------------------------------------------------------------------
    # RO path: zero-copy import committed weights into model params
    # ------------------------------------------------------------------

    def get_source_identity(self) -> Optional[SourceIdentity]:
        """Return the writer's committed :class:`SourceIdentity`, if available.

        An RO reader uses this to verify, before :meth:`materialize_module`,
        that the writer's layout matches its own.

        Returns:
            The writer's committed identity, or ``None`` when none is
            available (the caller then proceeds without a pre-materialize
            compatibility check).
        """
        # TODO(SOURCE-IDENTITY/GMS): persist the writer's serialized identity
        # in finalize_write and read it back here once the pool exposes a
        # metadata channel. This is the single seam the RO gate depends on.
        return None

    def materialize_module(self, model: nn.Module) -> None:
        """Zero-copy import committed weights into model params (RO path).

        Replaces meta-initialized parameters with CUDA tensors backed
        by GPU pointers from the shared memory region — no data copies,
        no disk I/O, just CUDA VMM remapping. The model's submodule
        layout must already match the writer's at commit time, including
        any aliases / derived buffers introduced by ``post_load_weights``.

        Args:
            model: The ``nn.Module`` to materialize. Walks the full
                module tree (including submodules like ``draft_model``
                added for speculative decoding) and rebinds matching
                parameters to GMS-backed storage.

        Raises:
            RuntimeError: If ``connect()`` has not been called yet.

        Note:
            ``post_load_weights()`` must be called on the model BEFORE
            this method. The order ensures that any aliases / derived
            parameters created by post-load hooks are present on the
            module tree at materialization time, so they are bound to
            the same GMS storage as their primary tensor.
        """
        if self._client is None:
            raise RuntimeError("GMS client not connected. Call connect() first.")

        from gpu_memory_service.client.torch.module import materialize_module_from_gms

        materialize_module_from_gms(self._client, model, device_index=self._device_index)

        logger.info(
            "GMS RO: materialized weights from %s (tag=%s, tp_rank=%d/%d, total_bytes=%.2f GiB)",
            self._socket_path,
            self._tag,
            self._mapping.tp_rank,
            self._mapping.tp_size,
            int(self._client.total_bytes) / (1 << 30),
        )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release the GMS session and evict it from the per-tag registry.

        Idempotent: a second call after a successful cleanup is a no-op
        because the client handle is dropped. Best-effort: any failure
        in upstream ``client.close()`` or ``evict_gms_client_memory_manager``
        is logged (warning for the outer error, debug for ``close()``)
        and swallowed so callers — including ``__del__`` paths in
        ``ModelLoader`` and ``PyTorchModelEngine`` — never raise from
        teardown.

        After return:
            - The local client handle is unset (``_client is None``).
            - A subsequent :meth:`connect` may re-establish a fresh
              session against the same daemon.
            - GMS-backed CUDA mappings remain alive on-device for any
              other process holding an RO lock on this ``tag``.
        """
        if self._client is None:
            return
        try:
            from gpu_memory_service.client.torch.allocator import evict_gms_client_memory_manager

            client = self._client
            try:
                client.close()
            except Exception as e:
                # Best-effort: even if close() fails, evict so a future
                # connect() in the same process can re-establish state.
                # Log at debug so unrelated shutdown noise doesn't drown
                # out the warning we already emit for the outer try.
                logger.debug("GMS client.close() failed (best-effort): %s", e)
            evict_gms_client_memory_manager(client)
            logger.info("GMS: disconnected from %s", self._socket_path)
        except Exception as e:
            logger.warning("GMS cleanup error: %s", e)
        finally:
            self._client = None
            self._is_rw = None


def _ptr_in_gms(gms_client, ptr: int) -> bool:
    """Whether a raw CUDA pointer falls inside an existing GMS mapping.

    Mirrors ``gpu_memory_service.integrations.trtllm.model_loader._ptr_in_gms``.
    Tolerates absence of the ``_mappings`` attribute on older GMS
    releases by returning ``False`` (treating the pointer as "untracked"
    so the caller's stray-handling path runs).

    Args:
        gms_client: A connected ``GMSClientMemoryManager``.
        ptr: A raw CUDA virtual address (``tensor.data_ptr()`` or a
            storage base pointer).

    Returns:
        ``True`` if ``ptr`` lies inside any mapping the client tracks
        for any tag, ``False`` otherwise. Never raises.
    """
    mappings = getattr(gms_client, "_mappings", None)
    if not mappings:
        return False
    for mapping in mappings.values():
        base = int(getattr(mapping, "va", 0))
        size = int(getattr(mapping, "size", 0))
        if base and size and base <= ptr < base + size:
            return True
    return False


def _storage_nbytes(tensor: torch.Tensor) -> int:
    """Bytes owned by ``tensor``'s underlying storage, including any padding.

    Used to size GMS mappings created in :meth:`GMSBackend.move_untracked_params`
    so the replacement allocation matches the original storage exactly,
    not the (possibly smaller) view the tensor exposes.

    Args:
        tensor: Any CUDA ``torch.Tensor``. View-vs-base distinction is
            intentional: we always want the whole storage size.

    Returns:
        Total bytes of the underlying ``UntypedStorage``.
    """
    storage = tensor.untyped_storage()
    return int(storage.nbytes())
