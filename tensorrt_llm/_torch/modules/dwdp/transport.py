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

"""DWDP Transport layer: MNNVL handle allocation and cross-process exchange.

DWDPTransport handles the setup-time work of:
1. Allocating MNNVL fabric handles for each local expert weight chunk.
2. Copying local parameter data into the fabric handle memory.
3. Exchanging handle bytes via an MPI communicator so all peers can import them.
4. Importing peer handles and creating read-only tensor views for P2P reads.

After create() completes, the caller obtains:
- MnnvlHandleSet: local handles for WeightBuffer to map into its composite VA.
- peer_views: immutable tensor views into every peer's MNNVL memory, used by
  WeightManager to read remote expert data via P2P.

Design:
- Interleaved allocation: each (layer, weight) is processed one at a time.
  After copying data into the MNNVL handle the original parameter is freed
  immediately to avoid OOM on memory-constrained systems.
- GB200 constraint: all handles use CU_MEM_HANDLE_TYPE_FABRIC.
- Peer views are immutable: after setup, MNNVL handles are never written again.
  P2P reads are safe across processes without synchronization.
"""

from __future__ import annotations

import ctypes
import os
from typing import Dict, List, Tuple, Union

import torch

from tensorrt_llm.logger import logger

try:
    from cuda.bindings import driver as cuda

    logger.debug("[DWDP transport] Using cuda.bindings.driver")
except ImportError:
    from cuda import cuda

    logger.debug("[DWDP transport] Falling back to legacy `cuda` bindings")

from .specs import LayerWeightSpecs, MnnvlHandleSet, PeerRanges, compute_peer_ranges
from .vmm import (
    align_down,
    align_up,
    check_cu_result,
    create_fabric_handle,
    free_va,
    get_allocation_granularity,
    map_handle,
    peer_handle_type,
    release_handle,
    reserve_va,
    set_access,
    tensor_from_ptr,
    unmap_va,
)

# Linux ``pidfd_open(2)`` and ``pidfd_getfd(2)`` syscall numbers (x86_64 /
# aarch64 share the same numbers).  Used on the POSIX_FILE_DESCRIPTOR path
# to dup an FD from a sibling DWDP MPI worker into the local fd table so
# ``cuMemImportFromShareableHandle(fd, POSIX_FILE_DESCRIPTOR)`` accepts it.
# Mirrors ``MnnvlMemory.open_mnnvl_memory`` in
# ``tensorrt_llm/_mnnvl_utils.py``.
_SYS_pidfd_open = 434
_SYS_pidfd_getfd = 438


def _pidfd_open(pid: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(_SYS_pidfd_open, pid, 0)
    if fd < 0:
        err = ctypes.get_errno()
        raise RuntimeError(f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}")
    return fd


def _pidfd_getfd(pidfd: int, remote_fd: int) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    local_fd = libc.syscall(_SYS_pidfd_getfd, pidfd, remote_fd, 0)
    if local_fd < 0:
        err = ctypes.get_errno()
        msg = (
            f"pidfd_getfd(pidfd={pidfd}, fd={remote_fd}) failed with errno "
            f"{err}: {os.strerror(err)}."
        )
        if err == 1:  # EPERM
            msg += (
                " Permission denied. If running in a container, try adding "
                "--cap-add=SYS_PTRACE to your docker run command."
            )
        else:
            msg += " This may be due to kernel version (requires Linux 5.6+)."
        raise RuntimeError(msg)
    return local_fd


def _export_handle(handle: int, htype: cuda.CUmemAllocationHandleType) -> Union[bytes, int]:
    """Export a peer-shareable handle.

    For ``CU_MEM_HANDLE_TYPE_FABRIC`` the result is a ``bytes`` blob holding
    the ``CUmemFabricHandle`` struct.  For
    ``CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`` the result is the local
    file descriptor (``int``); the caller is responsible for closing it
    after every peer has dup-ed it via ``pidfd_getfd``.
    """
    exported = check_cu_result(cuda.cuMemExportToShareableHandle(handle, htype, 0))
    if htype == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC:
        if hasattr(exported, "data"):
            return bytes(exported.data)
        return bytes(exported)
    # POSIX_FILE_DESCRIPTOR: ``exported`` is the FD as int.
    return int(exported)


def _import_handle(payload: Union[bytes, int], htype: cuda.CUmemAllocationHandleType) -> int:
    """Import a peer-shareable handle previously produced by ``_export_handle``.

    For FABRIC ``payload`` is the bytes blob; for POSIX_FD ``payload`` is a
    *local* file descriptor (already dup-ed via ``pidfd_getfd``).
    """
    imported_handle = check_cu_result(cuda.cuMemImportFromShareableHandle(payload, htype))
    return int(imported_handle)


class DWDPTransport:
    """MNNVL handle allocation and cross-process exchange for DWDP.

    This class manages the full lifecycle of MNNVL fabric handles:
    allocation, data population, export/import via an MPI communicator,
    and creation of peer tensor views for P2P reads.

    Use the ``create()`` class method to construct an instance. After
    construction, ``get_handle_set()`` provides handles for WeightBuffer
    and ``get_peer_views()`` provides read-only tensor views for
    WeightManager.

    Attributes:
        dwdp_rank: This process's DWDP rank.
        dwdp_size: Total number of DWDP ranks.
        device_id: CUDA device ordinal.
        granularity: VMM page granularity in bytes.
    """

    __slots__ = (
        "_dwdp_rank",
        "_dwdp_size",
        "_device_id",
        "_granularity",
        "_handle_set",
        "_peer_views",
        "_peer_ranges",
        "_local_handles",
        "_imported_handles",
        "_peer_va_mappings",
        "_local_export_fds",
        "_released",
    )

    def __init__(
        self,
        dwdp_rank: int,
        dwdp_size: int,
        device_id: int,
        granularity: int,
        handle_set: MnnvlHandleSet,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
        peer_ranges: PeerRanges,
        local_handles: List[int],
        imported_handles: List[int],
        peer_va_mappings: List[Tuple[int, int]],
        local_export_fds: List[int],
    ) -> None:
        """Internal constructor. Use DWDPTransport.create() instead.

        Args:
            dwdp_rank: This process's DWDP rank.
            dwdp_size: Total number of DWDP ranks.
            device_id: CUDA device ordinal.
            granularity: VMM page granularity in bytes.
            handle_set: Local MNNVL handles for WeightBuffer.
            peer_views: Immutable tensor views into peer MNNVL memory.
            peer_ranges: Per-peer ``(local_start, local_end_capped)`` tuples
                indexed by DWDP rank.  The capped end truncates the tail
                rank's padding; consumers (lookup_owner, fill_edge_bytes,
                weight_manager) read these to find the actual owner of a
                given expert id.
            local_handles: Raw local handle integers (for cleanup).
            imported_handles: Raw imported handle integers (for cleanup).
            peer_va_mappings: List of (va, size) for peer VA regions (for cleanup).
        """
        self._dwdp_rank = dwdp_rank
        self._dwdp_size = dwdp_size
        self._device_id = device_id
        self._granularity = granularity
        self._handle_set = handle_set
        self._peer_views = peer_views
        self._peer_ranges = peer_ranges
        self._local_handles = local_handles
        self._imported_handles = imported_handles
        self._peer_va_mappings = peer_va_mappings
        self._local_export_fds = local_export_fds
        self._released = False

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        local_params: Dict[Tuple[int, str], torch.Tensor],
        comm,
        dwdp_rank: int,
        dwdp_size: int,
        device_id: int,
        local_start: int,
        local_end: int,
        num_experts_per_worker: int,
        num_prefetch_experts: int,
    ) -> DWDPTransport:
        """Allocate MNNVL handles, populate them, and exchange with all peers.

        This is the main entry point. Each of the ``dwdp_size`` processes calls
        this method concurrently. The method:

        1. For each (layer_idx, weight_name) — one at a time to limit peak
           memory usage — every rank in the DWDP communicator:
           a. Allocates a fabric handle large enough for the local chunk.
           b. Maps the handle to a temporary VA and copies the local parameter
              tensor into it at the correct ``data_offset``.
           c. Frees the original parameter tensor immediately.
           d. Unmaps the temporary VA (the physical handle persists).
           e. Exports the handle to bytes and shares it with every peer via
              a per-pair ``comm.allgather(handle_bytes)``.  The allgather is
              itself a synchronization point so no explicit barrier is needed
              between Phase 1 iterations.
        2. For each peer rank (skipping self), for each (layer_idx, weight_name):
           a. Retrieves the peer's exported handle bytes from the Phase 1
              allgather cache.
           b. Imports the fabric handle.
           c. Maps it to a VA and creates a tensor view for P2P reads.
        3. ``comm.Barrier()`` after all imports — ensures no rank starts
           consuming peer views before every rank has finished importing.

        Args:
            layer_weight_specs: Per-layer weight specifications. Keys are layer
                indices, values are dicts mapping weight names to WeightSpec.
            local_params: Local parameter tensors keyed by (layer_idx, name).
                Each tensor has shape == spec.chunk_shape and is on the correct
                CUDA device. These tensors are consumed (freed) during the call.
            comm: mpi4py communicator scoped to the DWDP group (created by
                DwdpManager).  Must have exactly ``dwdp_size`` ranks.
            dwdp_rank: This process's DWDP rank (0..dwdp_size-1).
            dwdp_size: Total number of DWDP ranks in ``comm``.
            device_id: CUDA device ordinal.
            local_start: First local expert index (inclusive).
            local_end: Last local expert index (exclusive).

        Returns:
            Ready-to-use DWDPTransport with handles and peer views.

        Raises:
            ValueError: If local_params keys do not match layer_weight_specs.
            RuntimeError: If any CUDA operation fails.
        """
        granularity = get_allocation_granularity(device_id)
        htype = peer_handle_type()
        is_posix_fd = (
            htype == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )

        # POSIX_FD path: dup peer FDs into our fd table via pidfd_getfd.
        # We open one pidfd per peer ≠ self once up front and close them
        # after Phase 3 Barrier.  ``peer_pidfds`` is empty on the FABRIC
        # path (aarch64 / GB200).
        peer_pidfds: Dict[int, int] = {}
        # FDs returned by ``cuMemExportToShareableHandle`` on this rank.
        # Must outlive every peer's ``pidfd_getfd`` (Phase 2) but are
        # closed right after the Phase 3 Barrier — at that point every
        # peer has finished dup-ing them and the FDs no longer back
        # anything we need.  ``cuMemImportFromShareableHandle`` dups the
        # peer-side FD internally, so we do NOT have to track imported
        # FDs across phases: each one is closed inline immediately after
        # the import call returns, capping fd usage at ~num_handles
        # rather than ``dwdp_size × num_handles``.  The list is empty on
        # the FABRIC path (aarch64 / GB200).
        local_export_fds: List[int] = []

        # Pick any spec to read the model's total expert count.  All layers
        # share the same ``num_experts`` (the gate-side global expert table).
        first_spec = next(iter(next(iter(layer_weight_specs.values())).values()))
        num_experts_total = first_spec.num_experts

        # Compute every peer's valid expert range deterministically from the
        # shared DwdpConfig.  Used by lookup_owner downstream
        # (fill_edge_bytes, weight_manager.prefetch_layer, fixup-side
        # allgathers) so the same first-match owner policy is applied
        # everywhere — including non-uniform tail padding and redundancy
        # where adjacent ranges overlap.
        peer_ranges = compute_peer_ranges(
            dwdp_size=dwdp_size,
            num_experts_per_worker=num_experts_per_worker,
            num_prefetch_experts=num_prefetch_experts,
            num_experts_total=num_experts_total,
        )

        # Validate that every (layer_idx, name) in specs has a matching param
        expected_keys = set()
        for layer_idx, weight_specs in layer_weight_specs.items():
            for name in weight_specs:
                expected_keys.add((layer_idx, name))
        provided_keys = set(local_params.keys())
        if expected_keys != provided_keys:
            missing = expected_keys - provided_keys
            extra = provided_keys - expected_keys
            raise ValueError(f"local_params keys mismatch. Missing: {missing}, Extra: {extra}")

        handles: Dict[Tuple[int, str], int] = {}
        sizes: Dict[Tuple[int, str], int] = {}
        local_handles: List[int] = []
        imported_handles: List[int] = []
        peer_va_mappings: List[Tuple[int, int]] = []
        peer_views: Dict[Tuple[int, int, str], torch.Tensor] = {}

        # Per-pair allgather result cache: (layer_idx, name) -> list[payload]
        # indexed by peer rank.  Populated in Phase 1, consumed in Phase 2.
        # Payload is ``bytes`` on FABRIC and ``int`` (FD) on POSIX_FD.
        all_exports: Dict[Tuple[int, str], List[Union[bytes, int]]] = {}

        try:
            # On POSIX_FD: open one pidfd per peer ≠ self before Phase 1.
            if is_posix_fd:
                all_pids = comm.allgather(os.getpid())
                for peer_rank, peer_pid in enumerate(all_pids):
                    if peer_rank == dwdp_rank:
                        continue
                    peer_pidfds[peer_rank] = _pidfd_open(peer_pid)

            # ----------------------------------------------------------
            # Phase 1: Allocate local handles and allgather bytes
            # ----------------------------------------------------------
            # Every rank iterates pairs in the same sorted order so that
            # per-pair comm.allgather calls match across ranks.
            for layer_idx in sorted(layer_weight_specs.keys()):
                weight_specs = layer_weight_specs[layer_idx]
                for name in sorted(weight_specs.keys()):
                    spec = weight_specs[name]
                    key = (layer_idx, name)
                    local_param = local_params[key]

                    # Compute physical handle size using the SAME formula
                    # as PageAlignedLayout.compute() to guarantee that
                    # phys_size == layout.mnnvl_size.  On GB200, fabric
                    # handles require cuMemMap with size == phys_size
                    # (partial mapping returns CUDA_ERROR_NOT_SUPPORTED).
                    local_start_bytes = local_start * spec.expert_bytes
                    local_end_bytes = local_end * spec.expert_bytes
                    page_start = align_down(local_start_bytes, granularity)
                    page_end = align_up(local_end_bytes, granularity)
                    phys_size = page_end - page_start
                    data_offset = local_start_bytes - page_start

                    # (a) Create the fabric handle
                    handle = create_fabric_handle(phys_size, device_id)
                    local_handles.append(handle)
                    handles[key] = handle
                    sizes[key] = phys_size

                    # (b) Reserve temp VA, map, set access, create tensor view
                    temp_va = reserve_va(phys_size, granularity)
                    mapped = False
                    try:
                        map_handle(temp_va, phys_size, handle, 0)
                        mapped = True
                        set_access(temp_va, phys_size, device_id)

                        # Create a flat tensor view at data_offset within the
                        # mapped VA, sized exactly to the chunk.
                        data_ptr = temp_va + data_offset
                        handle_tensor = tensor_from_ptr(
                            ptr=data_ptr,
                            shape=spec.chunk_shape,
                            dtype=spec.dtype,
                            device_id=device_id,
                        )

                        # (c) Copy local param into the handle memory
                        handle_tensor.copy_(local_param)
                        torch.cuda.synchronize(device_id)

                        # (d) Free the original param to reclaim memory
                        local_param.untyped_storage().resize_(0)
                        torch.cuda.empty_cache()

                        # (e) Unmap temp VA (handle stays alive)
                        unmap_va(temp_va, phys_size)
                        mapped = False
                    finally:
                        if mapped:
                            try:
                                unmap_va(temp_va, phys_size)
                            except Exception as e:
                                logger.warning(
                                    f"[DWDPTransport] temp VA unmap on error "
                                    f"path failed (ignored): {e!r}"
                                )
                        free_va(temp_va, phys_size)

                    # (f) Export and allgather to all peers
                    exported = _export_handle(handle, htype)
                    if is_posix_fd:
                        # Track local FD so it can be closed at release time.
                        # It must remain open through Phase 2 so peers can
                        # pidfd_getfd it.
                        local_export_fds.append(int(exported))
                    # comm.allgather implicitly synchronizes all ranks on this
                    # (layer_idx, name) — no explicit barrier required.
                    all_exports[key] = comm.allgather(exported)

                    logger.debug(
                        f"[DWDPTransport] Rank {dwdp_rank}: exported + allgathered "
                        f"handle for layer={layer_idx} name={name} "
                        f"phys_size={phys_size} data_offset={data_offset}"
                    )

            logger.info(
                f"[DWDPTransport] Rank {dwdp_rank}: all handles exported, importing peer handles..."
            )

            # ----------------------------------------------------------
            # Phase 2: Import peer handles and create tensor views
            # ----------------------------------------------------------
            for peer_rank in range(dwdp_size):
                if peer_rank == dwdp_rank:
                    continue

                # Compute peer's expert range using the same config-driven
                # formula every rank uses for itself (size = num_experts_per_worker,
                # stride = num_prefetch_experts).  All ranks share the same
                # DwdpConfig, so peer ranges are deterministic — no allgather needed.
                for layer_idx in sorted(layer_weight_specs.keys()):
                    weight_specs = layer_weight_specs[layer_idx]
                    for name in sorted(weight_specs.keys()):
                        spec = weight_specs[name]

                        peer_local_start = peer_rank * num_prefetch_experts
                        peer_local_end = peer_local_start + num_experts_per_worker

                        peer_start_bytes = peer_local_start * spec.expert_bytes
                        peer_end_bytes = peer_local_end * spec.expert_bytes
                        peer_page_start = align_down(peer_start_bytes, granularity)
                        peer_page_end = align_up(peer_end_bytes, granularity)
                        phys_size = peer_page_end - peer_page_start
                        peer_data_offset = peer_start_bytes - peer_page_start

                        # (a) Retrieve peer handle payload from Phase 1 cache.
                        peer_payload = all_exports[(layer_idx, name)][peer_rank]

                        # (b) Import the handle.  On POSIX_FD we first dup
                        # the peer's FD into our fd table via pidfd_getfd;
                        # cuMemImportFromShareableHandle then accepts that
                        # local FD and dups it internally, so we close
                        # ours immediately after the import returns.  On
                        # FABRIC the bytes blob is consumed directly.
                        if is_posix_fd:
                            local_fd = _pidfd_getfd(peer_pidfds[peer_rank], int(peer_payload))
                            try:
                                imported_handle = _import_handle(local_fd, htype)
                            finally:
                                try:
                                    os.close(local_fd)
                                except OSError as e:
                                    logger.warning(
                                        f"[DWDPTransport] Failed to close "
                                        f"imported FD {local_fd}: {e}"
                                    )
                        else:
                            imported_handle = _import_handle(peer_payload, htype)
                        imported_handles.append(imported_handle)

                        # (c) Reserve VA, map, set access
                        peer_va = reserve_va(phys_size, granularity)
                        map_handle(peer_va, phys_size, imported_handle, 0)
                        set_access(peer_va, phys_size, device_id)
                        peer_va_mappings.append((peer_va, phys_size))

                        peer_data_ptr = peer_va + peer_data_offset
                        peer_tensor = tensor_from_ptr(
                            ptr=peer_data_ptr,
                            shape=spec.chunk_shape,
                            dtype=spec.dtype,
                            device_id=device_id,
                        )

                        view_key = (peer_rank, layer_idx, name)
                        peer_views[view_key] = peer_tensor

                        logger.debug(
                            f"[DWDPTransport] Rank {dwdp_rank}: imported "
                            f"peer={peer_rank} layer={layer_idx} name={name} "
                            f"peer_data_offset={peer_data_offset}"
                        )

            # ----------------------------------------------------------
            # Phase 3: Barrier — all handles imported
            # ----------------------------------------------------------
            comm.Barrier()

            # POSIX_FD: close the per-peer pidfds — they're no longer
            # needed once every imported handle has been created.
            for pidfd in peer_pidfds.values():
                try:
                    os.close(pidfd)
                except OSError as e:
                    logger.warning(f"[DWDPTransport] Failed to close pidfd {pidfd}: {e}")
            peer_pidfds.clear()

            # POSIX_FD: close our exported FDs.  They had to stay open
            # through Phase 2 so peers could ``pidfd_getfd`` them, but
            # the Barrier above guarantees every peer has finished doing
            # so.  Closing here (instead of at ``release()``) caps peak
            # fd usage during setup at ~num_handles per rank.
            for fd in local_export_fds:
                try:
                    os.close(fd)
                except OSError as e:
                    logger.warning(f"[DWDPTransport] Failed to close local export FD {fd}: {e}")
            local_export_fds.clear()

            logger.info(
                f"[DWDPTransport] Rank {dwdp_rank}: setup complete. "
                f"{len(handles)} local handles, "
                f"{len(peer_views)} peer views."
            )

            handle_set = MnnvlHandleSet(handles=handles, sizes=sizes)

            return cls(
                dwdp_rank=dwdp_rank,
                dwdp_size=dwdp_size,
                device_id=device_id,
                granularity=granularity,
                handle_set=handle_set,
                peer_views=peer_views,
                peer_ranges=peer_ranges,
                local_handles=local_handles,
                imported_handles=imported_handles,
                peer_va_mappings=peer_va_mappings,
                local_export_fds=local_export_fds,
            )

        except Exception as exc:
            # On failure, log the original exception then clean up everything
            # we allocated so far.
            logger.error(
                f"[DWDPTransport] Rank {dwdp_rank}: create() failed with {exc!r}; "
                f"cleaning up partial allocations"
            )
            for pidfd in peer_pidfds.values():
                try:
                    os.close(pidfd)
                except OSError:
                    pass
            _cleanup_resources(
                local_handles=local_handles,
                imported_handles=imported_handles,
                peer_va_mappings=peer_va_mappings,
                local_export_fds=local_export_fds,
            )
            raise

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_handle_set(self) -> MnnvlHandleSet:
        """Return the local MNNVL handle set for WeightBuffer.

        Each entry maps (layer_idx, weight_name) to a CUDA fabric handle
        containing this rank's local expert chunk.

        Returns:
            MnnvlHandleSet with handles and physical sizes.

        Raises:
            RuntimeError: If transport has been released.
        """
        if self._released:
            raise RuntimeError("DWDPTransport has been released")
        return self._handle_set

    def get_peer_views(self) -> Dict[Tuple[int, int, str], torch.Tensor]:
        """Return immutable P2P tensor views into every peer's MNNVL memory.

        Keys are (peer_rank, layer_idx, weight_name). Each tensor has
        shape == spec.chunk_shape and is read-only (writes would corrupt
        the peer's data).

        Returns:
            Dict mapping (peer_rank, layer_idx, name) to tensor views.

        Raises:
            RuntimeError: If transport has been released.
        """
        if self._released:
            raise RuntimeError("DWDPTransport has been released")
        return self._peer_views

    def get_peer_ranges(self) -> PeerRanges:
        """Return per-peer ``(local_start, local_end_capped)`` tuples.

        Used by ``fill_edge_bytes`` and the runtime ``WeightManager`` to
        resolve the owner of any given expert id under non-uniform
        partition or redundancy.  The end is capped at ``num_experts``
        (i.e., reflects the *valid* expert range — not the storage range
        which may include tail-padding slots).

        Raises:
            RuntimeError: If transport has been released.
        """
        if self._released:
            raise RuntimeError("DWDPTransport has been released")
        return self._peer_ranges

    @property
    def dwdp_rank(self) -> int:
        """This process's DWDP rank."""
        return self._dwdp_rank

    @property
    def dwdp_size(self) -> int:
        """Total number of DWDP ranks."""
        return self._dwdp_size

    @property
    def device_id(self) -> int:
        """CUDA device ordinal."""
        return self._device_id

    @property
    def granularity(self) -> int:
        """VMM page granularity in bytes."""
        return self._granularity

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release all CUDA VMM resources. Idempotent.

        This unmaps all peer VA regions, releases imported handles, and
        releases local handles. After this call, ``get_handle_set()`` and
        ``get_peer_views()`` will raise RuntimeError.
        """
        if self._released:
            return
        self._released = True

        # Invalidate tensor views (they point to VA we are about to free)
        self._peer_views.clear()

        _cleanup_resources(
            local_handles=self._local_handles,
            imported_handles=self._imported_handles,
            peer_va_mappings=self._peer_va_mappings,
            local_export_fds=self._local_export_fds,
        )

        self._local_handles.clear()
        self._imported_handles.clear()
        self._peer_va_mappings.clear()
        self._local_export_fds.clear()

        logger.debug(f"[DWDPTransport] Rank {self._dwdp_rank}: released all resources")

    def __del__(self) -> None:
        """Clean up on garbage collection (best-effort; errors logged to debug)."""
        try:
            self.release()
        except Exception as e:
            logger.debug(f"[DWDPTransport] __del__ release failed (ignored): {e!r}")

    def __enter__(self) -> DWDPTransport:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit — release resources."""
        self.release()
        return False


def _cleanup_resources(
    local_handles: List[int],
    imported_handles: List[int],
    peer_va_mappings: List[Tuple[int, int]],
    local_export_fds: List[int],
) -> None:
    """Best-effort cleanup of CUDA resources.

    Unmap and free all peer VA regions, release imported handles, then
    release local handles.  Errors are logged but not raised so that
    cleanup proceeds as far as possible.

    The POSIX_FD path releases its FDs eagerly during ``create()`` —
    imported FDs are closed inline right after
    ``cuMemImportFromShareableHandle`` (which dups them internally), and
    local export FDs are closed right after the Phase 3 Barrier — so by
    the time normal-path ``release()`` runs ``local_export_fds`` is
    already empty.  This function still iterates it because the
    error-path caller invokes us with whatever FDs were tracked at the
    moment of failure, which may include in-flight export FDs that the
    Phase 3 cleanup never reached.

    Args:
        local_handles: Local MNNVL handle integers.
        imported_handles: Imported MNNVL handle integers.
        peer_va_mappings: List of (va, size) for peer VA regions.
        local_export_fds: POSIX FDs returned by cuMemExportToShareableHandle
            that have not yet been closed (POSIX_FD path only).
    """
    # Step 1: Unmap and free peer VA regions
    for va, size in peer_va_mappings:
        try:
            unmap_va(va, size)
        except Exception as e:
            logger.warning(f"[DWDPTransport] Failed to unmap peer VA 0x{va:x}: {e}")
        try:
            free_va(va, size)
        except Exception as e:
            logger.warning(f"[DWDPTransport] Failed to free peer VA 0x{va:x}: {e}")

    # Step 2: Release imported handles
    for h in imported_handles:
        try:
            release_handle(h)
        except Exception as e:
            logger.warning(f"[DWDPTransport] Failed to release imported handle {h}: {e}")

    # Step 3: Release local handles
    for h in local_handles:
        try:
            release_handle(h)
        except Exception as e:
            logger.warning(f"[DWDPTransport] Failed to release local handle {h}: {e}")

    # Step 4: Defensively close any local export FDs still tracked (only
    # populated on the POSIX_FD failure path; empty on the normal path
    # since Phase 3 already closed them).
    for fd in local_export_fds:
        try:
            os.close(fd)
        except OSError as e:
            logger.warning(f"[DWDPTransport] Failed to close local export FD {fd}: {e}")
