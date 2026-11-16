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
"""The sharded row store of dense DWDP.

Every rank holds its row of every ``(layer, segment)`` blob in one handle per store chunk (one
handle unless ``DenseConfig.store_chunk_mb`` is set); every rank maps every peer's handles
read-only.  The store is written once in setup P2 and never changes afterwards; the runtime reads
it only through the copy plans.

* :class:`FabricShardStore` -- FABRIC handles exported over the DWDP group (aarch64 / NVL72).
* :class:`LoopbackShardStore` -- all ``N`` ranks' rows in this process, for single-process tests
  of the whole path (collector, rows, gather, repoint).

Addressing: rank ``p``'s row of ``(layer, segment)`` starts at
``chunk_bases()[p][segment.chunk] + segment.chunk_offset`` and is ``segment.row_bytes`` long;
``chunk_bases()[rank]`` is this rank's own, writable mapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from tensorrt_llm.logger import logger

from .. import transport, vmm
from .layout import row_pieces
from .types import LOG_PREFIX, ROW_ALIGN, DenseLayout, DenseSetupError

#: What one rank contributes to collective #2: one exported handle per store chunk, or None for
#: the loopback store.  Must be picklable.
StorePayload = list[bytes] | None

_FABRIC = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC


class DenseShardStore(ABC):
    """This rank's rows plus a read-only view of every peer's rows.

    Call order (setup): :meth:`allocate` and :meth:`write_local_rows` in P2, then
    ``synchronize``, then :meth:`export` (collective #2), then :meth:`import_peers` in P3, after
    which :meth:`chunk_bases` is complete.  :meth:`release` is valid at any point and idempotent.

    Each step runs at most once: calling :meth:`allocate` or :meth:`import_peers` a second time
    (also after one that raised part way) raises ``RuntimeError``, and so does every call except
    :meth:`release` after :meth:`release`.  Whatever a failed step created is freed by
    :meth:`release`.

    Args:
        layout: The shared layout (chunks and rows); the collected anchors supply the bytes.
        rank: This rank.
        device: The dense CUDA device (CPU is allowed for the loopback store in unit tests).

    Raises:
        ValueError: ``rank`` is outside ``[0, N)``, or a segment's rows do not fit its chunk at a
            ``ROW_ALIGN``-aligned offset.
    """

    def __init__(self, layout: DenseLayout, rank: int, device: torch.device) -> None:
        num_ranks = layout.num_ranks
        if not 0 <= rank < num_ranks:
            raise ValueError(f"rank must be in [0, {num_ranks}), got {rank}")
        for segment in layout.segments.values():
            if (
                not 0 <= segment.chunk < len(layout.chunks)
                or segment.chunk_offset < 0
                or segment.chunk_offset % ROW_ALIGN
                or segment.chunk_offset + segment.row_bytes > layout.chunks[segment.chunk].nbytes
            ):
                raise ValueError(
                    f"the rows of ({segment.layer}, {segment.segment.value}) do not fit store "
                    f"chunk {segment.chunk} at offset {segment.chunk_offset} "
                    f"(S={segment.row_bytes}); chunk offsets must be {ROW_ALIGN}-aligned"
                )
        self._layout = layout
        self._rank = rank
        self._num_ranks = num_ranks
        self._device = torch.device(device)
        self._allocate_started = False
        self._allocated = False
        self._import_started = False
        self._bases: tuple[tuple[int, ...], ...] | None = None
        # Set last: ``__del__`` releases nothing of an object whose constructor raised.
        self._released = False

    @abstractmethod
    def allocate(self) -> None:
        """P2: allocate and map the local chunks (``StoreChunk.handle_bytes`` each), zero-filled.

        Raises:
            RuntimeError: The allocation or mapping was refused (e.g. a FABRIC handle that large).
        """

    @abstractmethod
    def write_local_rows(self) -> None:
        """P2: copy this rank's row of every ``(layer, segment)`` into its local chunk.

        Bytes come from ``layout.row_pieces`` over the anchors' ``source``; everything else stays
        zero.  Asynchronous on the current stream; the caller synchronizes before :meth:`export`.
        """

    @abstractmethod
    def export(self) -> StorePayload:
        """P2: the payload peers need to import this rank's chunks (collective #2)."""

    @abstractmethod
    def import_peers(self, payloads: Sequence[StorePayload]) -> None:
        """P3: import and map every peer's chunks read-only.

        Args:
            payloads: Collective #2 results, indexed by rank (this rank's entry included).
        """

    @abstractmethod
    def chunk_bases(self) -> tuple[tuple[int, ...], ...]:
        """``bases[p][c]``: address of rank ``p``'s chunk ``c`` as mapped here; ``ROW_ALIGN``-aligned.

        Raises:
            RuntimeError: Before :meth:`import_peers`.
        """

    @abstractmethod
    def release(self) -> None:
        """Unmap and free the peer imports, then the local chunks; idempotent."""

    def __del__(self) -> None:
        """Best-effort :meth:`release` for owners that miss the explicit teardown.

        Must tolerate an object whose ``__init__`` raised part way, and must never raise.
        """
        if getattr(self, "_released", True):
            return
        try:
            self.release()
        except Exception as exc:  # a finalizer must not raise
            logger.debug(f"{LOG_PREFIX} store release from __del__ failed: {exc!r}")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _require_open(self, action: str) -> None:
        if self._released:
            raise RuntimeError(f"dense DWDP store: {action} called after release")

    def _begin_allocate(self) -> None:
        """Claim the single :meth:`allocate` of this store."""
        self._require_open("allocate")
        if self._allocate_started:
            raise RuntimeError("dense DWDP store: allocate already ran")
        self._allocate_started = True

    def _require_allocated(self, action: str) -> None:
        self._require_open(action)
        if not self._allocated:
            raise RuntimeError(f"dense DWDP store: {action} called before a successful allocate")

    def _begin_import(self, payloads: Sequence[StorePayload]) -> list[StorePayload]:
        """Claim the single :meth:`import_peers` and return the payloads as a list of ``N``."""
        self._require_allocated("import_peers")
        if self._import_started:
            raise RuntimeError("dense DWDP store: import_peers already ran")
        self._import_started = True
        gathered = list(payloads)
        if len(gathered) != self._num_ranks:
            raise ValueError(
                f"import_peers needs one payload per rank ({self._num_ranks}), got {len(gathered)}"
            )
        return gathered

    def _checked_bases(self) -> tuple[tuple[int, ...], ...]:
        self._require_open("chunk_bases")
        if self._bases is None:
            raise RuntimeError("dense DWDP store: chunk_bases called before import_peers")
        return self._bases

    def _check_sources(self) -> None:
        """Every anchor still holds its flat ``uint8`` source of the recorded size.

        Raises:
            DenseSetupError: A source was dropped (after commit) or does not match its anchor.
        """
        for segment in self._layout.segments.values():
            for anchor in segment.anchors:
                source = anchor.source
                if source is None:
                    raise DenseSetupError(
                        f"cannot write the store rows: the source of {anchor.path} was dropped"
                    )
                if (
                    source.dtype != torch.uint8
                    or source.dim() != 1
                    or source.numel() != anchor.nbytes
                ):
                    raise DenseSetupError(
                        f"the source of {anchor.path} is {source.dtype} {tuple(source.shape)}; "
                        f"expected a flat uint8 tensor of {anchor.nbytes} bytes"
                    )

    def _write_rows(self, row: int, chunks: Sequence[torch.Tensor]) -> None:
        """Copy row ``row`` of every ``(layer, segment)`` into ``chunks[c]`` at its chunk offset.

        Args:
            row: The row (rank) to write.
            chunks: Flat ``uint8`` tensors over the chunks of the store holding row ``row``.
        """
        for segment in self._layout.segments.values():
            target = chunks[segment.chunk]
            for piece in row_pieces(segment, row, num_ranks=self._num_ranks):
                source = segment.anchors[piece.anchor_index].source
                start = segment.chunk_offset + piece.row_offset
                target[start : start + piece.nbytes].copy_(
                    source[piece.anchor_offset : piece.anchor_offset + piece.nbytes]
                )

    def _synchronize_current_stream(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()


@dataclass(eq=False)
class _Mapping:
    """One VMM handle and its mapping; the fields record how far its setup got."""

    handle: int
    size: int
    va: int = 0
    mapped: bool = False


def _handle_size(handle_bytes: int, granularity: int) -> int:
    """Physical size of a chunk's handle: whole granules, at least one (an empty chunk too)."""
    return max(vmm.align_up(handle_bytes, granularity), granularity)


def _layer_span(layers: Sequence[int]) -> str:
    """How a store chunk's layers are named in messages."""
    if not layers:
        return "no layers"
    if len(layers) == 1:
        return f"layer {layers[0]}"
    return f"layers {layers[0]}..{layers[-1]}"


def _describe_payload(payload: object) -> str:
    if isinstance(payload, (list, tuple)):
        kinds = sorted({type(blob).__name__ for blob in payload})
        return f"a {type(payload).__name__} of {len(payload)} ({', '.join(kinds) or 'nothing'})"
    return type(payload).__name__


def _set_read_only_access(va: int, size: int, device_id: int) -> None:
    """``cuMemSetAccess`` with ``PROT_READ`` for ``device_id`` over ``[va, va + size)``."""
    desc = vmm.get_access_desc(device_id)
    desc.flags = vmm.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
    vmm.check_cu_result(vmm.cuda.cuMemSetAccess(va, size, [desc], 1))


class FabricShardStore(DenseShardStore):
    """Store backed by one FABRIC handle per chunk.

    Each chunk is a ``vmm.create_fabric_handle`` allocation, mapped locally for the P2 writes and
    as the copy source of the local row slices, exported like the expert transport's FABRIC
    handles and imported and mapped read-only on every peer.

    A chunk's handle is ``StoreChunk.handle_bytes`` rounded up to whole allocation granules, and
    at least one granule: a chunk without rows still gets a handle, so every chunk has a mapped
    base.  Each rank therefore allocates one granule more than ``DenseLayout.store_bytes`` counts
    for every chunk without rows and, on a device whose granularity exceeds ``STORE_ALIGN``, up to
    a granule more for every other chunk; :meth:`allocate` logs the difference.

    The local mapping is read-write; peer mappings grant ``PROT_READ`` only, so a stray write into
    a peer's rows faults instead of corrupting them.  Every rank maps each peer handle whole, with
    the size it computes from the shared layout.  Run it on the dense device's thread with that
    device's context current, as setup does.

    Raises:
        ValueError: ``device`` is not a CUDA device (and the base class checks).
    """

    def __init__(self, layout: DenseLayout, rank: int, device: torch.device) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"the FABRIC store needs a CUDA device, got {device}")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        super().__init__(layout, rank, device)
        self._device_id: int = device.index
        self._local: list[_Mapping] = []
        self._imports: list[_Mapping] = []
        self._views: list[torch.Tensor] = []
        self._exported: list[bytes] | None = None

    def allocate(self) -> None:
        """See :meth:`DenseShardStore.allocate`.

        Each chunk's handle is ``max(align_up(handle_bytes, granularity), granularity)`` bytes
        (see the class docstring).  The zero fill runs on the current stream, which is
        synchronized before returning.

        Raises:
            RuntimeError: The platform's peer handles are not FABRIC, a CUDA VMM call failed, or
                a local view does not alias its mapping.  A refused handle names its chunk, its
                size and its layers.  A launch keeps the store in one handle; tests can split it
                into smaller ones with ``DenseConfig.store_chunk_mb``.
        """
        self._begin_allocate()
        handle_type = vmm.peer_handle_type()
        if handle_type != _FABRIC:
            raise RuntimeError(
                f"the dense DWDP store needs FABRIC peer handles; this platform uses "
                f"{handle_type.name}"
            )
        granularity = vmm.get_allocation_granularity(self._device_id)
        chunks = self._layout.chunks
        for chunk in chunks:
            size = _handle_size(chunk.handle_bytes, granularity)
            try:
                handle = vmm.create_fabric_handle(size, self._device_id)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"cuMemCreate of the FABRIC handle of store chunk {chunk.index} of "
                    f"{len(chunks)} ({size} bytes, {_layer_span(chunk.layers)}) on "
                    f"cuda:{self._device_id} failed: {exc}"
                ) from exc
            record = _Mapping(handle=handle, size=size)
            self._local.append(record)
            record.va = vmm.reserve_va(size, granularity)
            vmm.map_handle(record.va, size, record.handle)
            record.mapped = True
            vmm.set_access(record.va, size, self._device_id)
            view = vmm.tensor_from_ptr(record.va, (size,), torch.uint8, self._device_id)
            if view.data_ptr() != record.va:
                raise RuntimeError(
                    f"the view of store chunk {chunk.index} at {record.va:#x} is a copy at "
                    f"{view.data_ptr():#x}; the store needs views that alias its mapping"
                )
            self._views.append(view)
            view.zero_()
        self._synchronize_current_stream()
        self._allocated = True
        allocated = sum(record.size for record in self._local)
        message = (
            f"{LOG_PREFIX} FABRIC store: rank {self._rank} allocated {allocated} bytes in "
            f"{len(self._local)} handle(s) on cuda:{self._device_id}"
        )
        if allocated == self._layout.store_bytes:
            logger.debug(message)
        else:
            logger.info(
                f"{message}, {allocated - self._layout.store_bytes} more than the layout's "
                f"store_bytes ({self._layout.store_bytes}): every handle is whole granules of "
                f"{granularity} bytes, at least one"
            )

    def write_local_rows(self) -> None:
        """See :meth:`DenseShardStore.write_local_rows`.

        Raises:
            RuntimeError: Before a successful :meth:`allocate`, or after :meth:`release`.
            DenseSetupError: An anchor's source was dropped or does not match its anchor.
        """
        self._require_allocated("write_local_rows")
        self._check_sources()
        self._write_rows(self._rank, self._views)

    def export(self) -> StorePayload:
        """See :meth:`DenseShardStore.export`.

        Returns:
            One exported FABRIC handle (``bytes``) per store chunk, in chunk order.

        Raises:
            RuntimeError: Before a successful :meth:`allocate`, after :meth:`release`, or the
                export failed.
        """
        self._require_allocated("export")
        if self._exported is None:
            self._exported = [
                transport._export_handle(record.handle, _FABRIC) for record in self._local
            ]
        return list(self._exported)

    def import_peers(self, payloads: Sequence[StorePayload]) -> None:
        """See :meth:`DenseShardStore.import_peers`.

        Peers are imported in ascending rank order, chunks in chunk order; this rank's entry
        must be what :meth:`export` returned and its local mappings serve as its bases.

        Raises:
            ValueError: Not one payload per rank, a payload that is not one exported handle per
                chunk, or this rank's entry differs from its export (payloads out of rank order).
            RuntimeError: Before :meth:`export`, on a second call, after :meth:`release`, a CUDA
                VMM call failed, or a mapping is not ``ROW_ALIGN``-aligned.
        """
        self._require_allocated("import_peers")
        if self._exported is None:
            raise RuntimeError("dense DWDP store: import_peers called before export")
        gathered = self._begin_import(payloads)
        num_chunks = len(self._local)
        for peer, payload in enumerate(gathered):
            if (
                not isinstance(payload, (list, tuple))
                or len(payload) != num_chunks
                or not all(isinstance(blob, bytes) and blob for blob in payload)
            ):
                raise ValueError(
                    f"rank {peer}'s store payload must be {num_chunks} exported FABRIC handle(s) "
                    f"(non-empty bytes), got {_describe_payload(payload)}"
                )
        if list(gathered[self._rank]) != self._exported:
            raise ValueError(
                f"payload {self._rank} is not this rank's export; the payloads must be indexed by "
                f"rank"
            )
        granularity = vmm.get_allocation_granularity(self._device_id)
        bases = []
        for peer, payload in enumerate(gathered):
            if peer == self._rank:
                bases.append(tuple(record.va for record in self._local))
                continue
            row = []
            for local, blob in zip(self._local, payload):
                record = _Mapping(handle=transport._import_handle(blob, _FABRIC), size=local.size)
                self._imports.append(record)
                record.va = vmm.reserve_va(record.size, granularity)
                vmm.map_handle(record.va, record.size, record.handle)
                record.mapped = True
                _set_read_only_access(record.va, record.size, self._device_id)
                row.append(record.va)
            bases.append(tuple(row))
        unaligned = [hex(base) for row in bases for base in row if base % ROW_ALIGN]
        if unaligned:
            raise RuntimeError(f"store mappings {unaligned[:4]} are not {ROW_ALIGN}-aligned")
        self._bases = tuple(bases)
        logger.debug(
            f"{LOG_PREFIX} FABRIC store: rank {self._rank} mapped {len(self._imports)} peer "
            f"chunk(s) read-only"
        )

    def chunk_bases(self) -> tuple[tuple[int, ...], ...]:
        """See :meth:`DenseShardStore.chunk_bases`.

        Raises:
            RuntimeError: Before a successful :meth:`import_peers`, or after :meth:`release`.
        """
        return self._checked_bases()

    def release(self) -> None:
        """See :meth:`DenseShardStore.release`.

        Order (as the expert transport): synchronize the device if anything was created, then
        unmap and free the VA of every peer mapping, release the imported handles, unmap and free
        the local mappings, and release the local handles.  Each step covers exactly what
        :meth:`allocate` and :meth:`import_peers` created, also after one of them raised part
        way.  Every step runs even if an earlier one fails; the first error is raised at the end
        and later ones are logged.

        Raises:
            RuntimeError: The first failed step's CUDA error; the store is released as far as
                possible.
        """
        if self._released:
            return
        self._released = True
        self._views = []
        self._bases = None
        imports, self._imports = self._imports, []
        local, self._local = self._local, []
        failures: list[tuple[str, Exception]] = []

        def attempt(step: str, call: Callable[..., object], *args: int) -> None:
            try:
                call(*args)
            except Exception as exc:  # every teardown step runs; the first failure is raised below
                failures.append((step, exc))

        if imports or local:
            attempt("synchronize", torch.cuda.synchronize, self._device_id)
        for records, kind in ((imports, "peer"), (local, "local")):
            for record in records:
                if record.mapped:
                    attempt(f"unmap {kind} VA {record.va:#x}", vmm.unmap_va, record.va, record.size)
                if record.va:
                    attempt(f"free {kind} VA {record.va:#x}", vmm.free_va, record.va, record.size)
            for record in records:
                attempt(f"release {kind} handle {record.handle}", vmm.release_handle, record.handle)
        if failures:
            for step, exc in failures[1:]:
                logger.error(f"{LOG_PREFIX} FABRIC store release: {step} also failed: {exc!r}")
            raise failures[0][1]
        logger.debug(
            f"{LOG_PREFIX} FABRIC store released: rank {self._rank}, {len(imports)} peer and "
            f"{len(local)} local handle(s)"
        )


class LoopbackShardStore(DenseShardStore):
    """All ``N`` ranks' rows in this process, as ``ROW_ALIGN``-aligned torch buffers.

    Every rank's weights are identical, so :meth:`write_local_rows` writes the rows of every rank
    from the local anchors; :meth:`export` returns None and :meth:`import_peers` only checks the
    payloads.  Used by the CPU setup tests.

    Rank ``p``'s chunk ``c`` is its own zero-filled allocation of ``StoreChunk.handle_bytes``
    bytes (plus ``ROW_ALIGN`` to align its base), so every rank's bases are distinct.
    """

    def __init__(self, layout: DenseLayout, rank: int, device: torch.device) -> None:
        super().__init__(layout, rank, device)
        self._chunks: list[list[torch.Tensor]] = []
        self._addresses: list[tuple[int, ...]] = []

    def allocate(self) -> None:
        """See :meth:`DenseShardStore.allocate`; allocates the chunks of all ``N`` ranks.

        On CUDA the zero fill runs on the current stream, which is synchronized before
        returning.
        """
        self._begin_allocate()
        for _ in range(self._num_ranks):
            chunks = []
            addresses = []
            for chunk in self._layout.chunks:
                storage = torch.zeros(
                    chunk.handle_bytes + ROW_ALIGN, dtype=torch.uint8, device=self._device
                )
                pad = (-storage.data_ptr()) % ROW_ALIGN
                chunks.append(storage[pad : pad + chunk.handle_bytes])
                addresses.append(storage.data_ptr() + pad)
            self._chunks.append(chunks)
            self._addresses.append(tuple(addresses))
        self._synchronize_current_stream()
        self._allocated = True

    def write_local_rows(self) -> None:
        """See :meth:`DenseShardStore.write_local_rows`; writes all ``N`` ranks' rows.

        Raises:
            RuntimeError: Before a successful :meth:`allocate`, or after :meth:`release`.
            DenseSetupError: An anchor's source was dropped or does not match its anchor.
        """
        self._require_allocated("write_local_rows")
        self._check_sources()
        for row, chunks in enumerate(self._chunks):
            self._write_rows(row, chunks)

    def export(self) -> StorePayload:
        """See :meth:`DenseShardStore.export`; always None.

        Raises:
            RuntimeError: Before a successful :meth:`allocate`, or after :meth:`release`.
        """
        self._require_allocated("export")
        return None

    def import_peers(self, payloads: Sequence[StorePayload]) -> None:
        """See :meth:`DenseShardStore.import_peers`; every payload must be None.

        Raises:
            ValueError: Not one payload per rank, or a payload that is not None (a peer with
                another kind of store).
            RuntimeError: Before a successful :meth:`allocate`, on a second call, or after
                :meth:`release`.
        """
        gathered = self._begin_import(payloads)
        foreign = [peer for peer, payload in enumerate(gathered) if payload is not None]
        if foreign:
            raise ValueError(
                f"the loopback store expects a None payload from every rank; ranks {foreign} "
                f"sent store handles"
            )
        self._bases = tuple(self._addresses)

    def chunk_bases(self) -> tuple[tuple[int, ...], ...]:
        """See :meth:`DenseShardStore.chunk_bases`.

        Raises:
            RuntimeError: Before a successful :meth:`import_peers`, or after :meth:`release`.
        """
        return self._checked_bases()

    def release(self) -> None:
        """See :meth:`DenseShardStore.release`; drops the buffers after a device synchronize."""
        if self._released:
            return
        self._released = True
        self._bases = None
        chunks, self._chunks = self._chunks, []
        self._addresses = []
        if chunks and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        del chunks
