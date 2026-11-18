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
"""The MNNVL memory transport of the host exchange, a ``HostExchangeGroup``.

In a round, each rank writes its slot into every active rank's segment and polls its own segment.
Each rank owns one segment with two slot sets and one slot per sender::

    segment: | slot set 0: slot 0 | slot 1 | ... | slot set 1: slot 0 | slot 1 | ... |
    slot:    | header | payload packet 1 | payload packet 2 | ... |

A slot is a run of 8-byte packets. The lower 32 bits hold the payload length in the header and 4
payload bytes in a payload packet. The upper 32 bits hold the round's epoch, so a packet is new when
it carries the current epoch and a slot needs no ready flag.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.internal.runtime import wait_for_slots
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div

from . import mnnvl_memory
from .membership import StoreGroup, has_rank, rank_mask
from .membership_mnnvl_memory import MembershipMnnvlMemory, comm_members
from .mnnvl_memory import (
    MnnvlMemory,
    TpMnnvlMemory,
    _check_cu_result,
    _MnnvlAllocationRecord,
    _MnnvlAllocationState,
)

# Rows sent by one copy as (destination, destination pitch, source, source pitch, width in bytes,
# height); a source pitch of 0 sends one source row to every destination row.
SlotCopy = tuple[int, int, int, int, int, int]


class HostExchangeMnnvlMemory(MembershipMnnvlMemory):
    """MNNVL memory of one row per rank, every row mapped in every process at ``ptr + r * stride``.

    The rows sit back to back, so one strided copy reaches every row. The row of an inactive rank
    maps local scratch memory. The rows a former member may still write to are remapped onto fresh
    pages, so its late writes land in pages that no rank maps.
    """

    absent_rows_on_scratch = True
    # Whether ``close`` ran, after which neither ``close`` nor collection releases the rows again.
    closed: bool = False

    @classmethod
    def uses_fabric_handles(cls, dev_id: int) -> bool:
        """Return whether device ``dev_id`` shares memory through fabric handles."""
        handle_type = cls.get_allocation_prop(dev_id).requestedHandleTypes
        return handle_type == mnnvl_memory.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC

    def __init__(self, comm: StoreGroup, nbytes: int):
        """Reserve a row of ``nbytes`` per member of ``comm`` and map it among the active members.

        The rows are shared over ``comm`` rather than over the communicator of a ``Mapping``, so
        this replaces ``MnnvlMemory.__init__`` and the rows have no ``mapping``.

        Raises:
            MnnvlPeerHandleError: An active rank could not import a peer's handle.
            MnnvlMappingError: Mapping or zeroing failed on some active ranks.
            RuntimeError: The device does not share memory through fabric handles.
        """
        cls = type(self)
        dev_id = cls._init_device()
        if not cls.uses_fabric_handles(dev_id):
            raise RuntimeError(
                "HostExchangeMnnvlMemory requires fabric memory handles, which this device lacks"
            )
        granularity = cls.get_allocation_granularity(dev_id)
        stride = granularity * ceil_div(nbytes, granularity)
        nranks, comm_rank, members = comm.Get_size(), comm.Get_rank(), comm_members(comm)
        ptr = int(
            _check_cu_result(
                mnnvl_memory.cuda.cuMemAddressReserve(nranks * stride, granularity, 0, 0)
            )
        )
        cls._register_allocation(
            ptr,
            _MnnvlAllocationRecord(
                comm=None,
                comm_size=nranks,
                comm_rank=comm_rank,
                comm_membership=members,
                aligned_size=stride,
                mem_handles=[None] * nranks,
                start_address=ptr,
                rank_stride=stride,
                address_offset=0,
                state=_MnnvlAllocationState.UNMAPPED,
            ),
        )
        self.segment_size = nbytes
        self.ptr, self.rank_stride = ptr, stride
        try:
            self.remap(comm, frozenset(members))
        except BaseException:
            self.close()
            raise

    def _reset_own_row(self) -> None:
        _check_cu_result(
            mnnvl_memory.cuda.cuMemsetD8(
                self.row(type(self).allocated_map[self.ptr].comm_rank), 0, self.stride
            )
        )
        torch.cuda.synchronize()

    @property
    def stride(self) -> int:
        return self.rank_stride

    def row(self, rank: int) -> int:
        return self.ptr + rank * self.rank_stride

    def close(self) -> None:
        """Unmap every row and free the address range, once even if that fails."""
        if self.closed:
            return
        self.closed = True
        type(self).close_mnnvl_memory(self.ptr)

    def __del__(self):
        if not self.closed:
            super().__del__()


class MnnvlHostExchangeGroup:
    """``HostExchangeGroup`` over MNNVL memory, one ``HostExchangeMnnvlMemory`` row per rank.

    A round runs the all-to-all kernels' LL protocol on the copy engines, so it takes no SMs. Round
    ``epoch`` uses slot set ``epoch & 1``, so a rank never overwrites a slot that a slower rank
    still reads. A rank completes a round that another rank finished from the peers' rows.
    """

    # The payload packets (1 KiB of payload) that the header poll copies with each header. A payload
    # that fits goes to every row, so it arrives with the header poll and needs no second poll.
    _HEADER_POLL_PACKETS = 256
    # A wait polls back to back for _SPIN_S, then sleeps between polls. _SPIN_S is longer than the
    # normal prefill imbalance between ranks, so a healthy step's wait sees a late rank's packets as
    # soon as they land, and a wait on a failed rank stops spinning soon after. An idle wait may last
    # until the next request arrives, so it sleeps after _IDLE_SPIN_S.
    _SPIN_S = 150e-3
    _IDLE_SPIN_S = 20e-6

    def __init__(
        self, rank: int, nranks: int, rows: HostExchangeMnnvlMemory, slot_payload_bytes: int
    ):
        self._cuda = mnnvl_memory.cuda
        self.rank = rank
        self.nranks = nranks
        self.rows = rows
        self.words = slot_payload_bytes // 4
        self._header_poll_packets = 1 + min(self.words, self._HEADER_POLL_PACKETS)
        self._slot_bytes = 8 * (1 + self.words)
        # A round uses slot set epoch & 1, which starts at (epoch & 1) * _slot_set_bytes.
        self._slot_set_bytes = nranks * self._slot_bytes
        # Ranks the running wait still lacks packets from, and the wake word that cancel() sets
        # from another thread to end the wait.
        self._waiting = np.zeros(nranks, dtype=bool)
        self._wake = np.zeros(1, dtype=np.int32)
        self._stream = torch.cuda.Stream()
        # The outgoing slot. Each send copies it to every row in a range with a source pitch of 0.
        self._send = self._pinned((1, 1 + self.words))
        self._recv = self._pinned((nranks, 1 + self.words))
        self._send_halves = self._send.view(np.uint32)[0]
        self._send_ptr = self._send.ctypes.data
        # The copies of the staged slot, which the next wait sends.
        self._sends: list[SlotCopy] = []
        # Cached _send_runs results, keyed by recipient mask.
        self._send_runs_by_recipients: dict[int, tuple[tuple[int, int, bool], ...]] = {}
        self._send_runs_to_all = ((0, nranks, True),)
        # The round that an install completed, kept until an install succeeds.
        self._completed: Optional[np.ndarray] = None

    @staticmethod
    def _pinned(shape: tuple[int, int]) -> np.ndarray:
        """Return a host array of uint64 packets, pinned where pinning helps."""
        return (
            torch.empty(shape, dtype=torch.int64, pin_memory=prefer_pinned())
            .numpy()
            .view(np.uint64)
        )

    @staticmethod
    def buffer_bytes(nranks: int, slot_payload_bytes: int) -> int:
        return 2 * nranks * 8 * (1 + slot_payload_bytes // 4)

    @classmethod
    def requirements(cls) -> list[tuple[bool, str]]:
        if not MnnvlMemory.supports_mnnvl():
            return [
                (
                    False,
                    "GPUs in one multi-node NVLink (MNNVL) domain, such as a GB200 NVL72, "
                    "because fault tolerance remaps MNNVL memory so that a failed rank's writes "
                    "cannot reach the active ranks",
                )
            ]
        return [
            (
                HostExchangeMnnvlMemory.uses_fabric_handles(torch.cuda.current_device()),
                "MNNVL memory shared through fabric handles (GB200-class aarch64 nodes "
                "with IMEX running and an IMEX channel in the container), because fault "
                "tolerance remaps MNNVL memory and exchanges host values through fabric handles",
            )
        ]

    @classmethod
    def open(cls, mapping: Mapping, slot_payload_bytes: int) -> tuple[MnnvlHostExchangeGroup, int]:
        comm = TpMnnvlMemory.get_comm(mapping)
        active_mask = rank_mask(
            (rank for rank, active in enumerate(comm.allgather(True)) if active), mapping.tp_size
        )
        rows = HostExchangeMnnvlMemory(comm, cls.buffer_bytes(mapping.tp_size, slot_payload_bytes))
        return cls(mapping.tp_rank, mapping.tp_size, rows, slot_payload_bytes), active_mask

    def _send_runs(self, recipients: int) -> tuple[tuple[int, int, bool], ...]:
        """Return ``(first rank, count, with payload)`` per run of ranks in or out of ``recipients``."""
        runs = self._send_runs_by_recipients.get(recipients)
        if runs is None:
            full = [has_rank(recipients, r) for r in range(self.nranks)]
            runs, first = [], 0
            for r in range(1, self.nranks + 1):
                if r == self.nranks or full[r] != full[first]:
                    runs.append((first, r - first, full[first]))
                    first = r
            runs = self._send_runs_by_recipients[recipients] = tuple(runs)
        return runs

    def start_round(
        self, epoch: int, words: np.ndarray, length: int, recipients: Optional[int]
    ) -> None:
        # The header goes to every row, the payload to the recipients' rows.
        packets = 1 + words.size
        halves = self._send_halves[: 2 * packets]
        halves[0] = length
        halves[1::2] = epoch
        halves[2::2] = words
        if recipients is None or words.size <= self._HEADER_POLL_PACKETS:
            # A payload that a header poll copies whole goes to every row.
            runs = self._send_runs_to_all
        else:
            runs = self._send_runs(recipients)
        own_slot = (epoch & 1) * self._slot_set_bytes + self.rank * self._slot_bytes
        self._sends = [
            (
                self.rows.row(first) + own_slot,
                self.rows.stride,
                self._send_ptr,
                0,
                8 * (packets if payload else 1),
                count,
            )
            for first, count, payload in runs
        ]

    def wait(
        self,
        epoch: int,
        active: np.ndarray,
        readers: np.ndarray,
        idle: bool = False,
        received: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        sends, self._sends = self._sends, []
        if received is not None:
            done = wait_for_slots(
                self._stream.cuda_stream,
                sends,
                None,
                received,
                active,
                readers,
                epoch,
                received.shape[1],
                self._waiting,
                self._wake,
                0.0,
            )
            return received if done else None
        if wait_for_slots(
            self._stream.cuda_stream,
            sends,
            self.rows.row(self.rank) + (epoch & 1) * self._slot_set_bytes,
            self._recv,
            active,
            readers,
            epoch,
            self._header_poll_packets if readers.any() else 1,
            self._waiting,
            self._wake,
            self._IDLE_SPIN_S if idle else self._SPIN_S,
        ):
            return self._recv
        # Clear the word before the caller checks why, so that a later cancel ends the next wait.
        self._wake[0] = 0
        return None

    def cancel(self) -> None:
        self._wake[0] = 1

    @property
    def waiting_for(self) -> int:
        # The running wait rewrites _waiting without the GIL. np.flatnonzero counts the set flags,
        # then fills their indices, so it needs a copy that cannot change in between.
        return rank_mask((int(r) for r in np.flatnonzero(self._waiting.copy())), self.nranks)

    def end_wait(self) -> None:
        self._waiting[:] = False

    def install(
        self,
        comm: StoreGroup,
        active_mask: int,
        incomplete: Optional[tuple[int, np.ndarray, np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        # The rows a former member may still write to get fresh pages.
        fresh_rows = self.rows.rows_needing_fresh_pages(comm)
        if incomplete is None:
            self._completed = None
        elif self._completed is None:
            # Read before the remap, as the remap may release the peers' rows.
            self._completed = self._finish_round(*incomplete, active_mask)
        self.rows.remap(comm, fresh_rows)
        completed, self._completed = self._completed, None
        return completed

    def _finish_round(
        self, epoch: int, readers: np.ndarray, round_active: np.ndarray, active_mask: int
    ) -> np.ndarray:
        """Complete this rank's round ``epoch`` with the slots it lacks, copied from peer rows.

        A reader's slot counts once it is whole. Another rank's slot counts once its header is in.
        """
        parity_offset = (epoch & 1) * self._slot_set_bytes
        rows = self._pinned((self.nranks, 1 + self.words))
        self._copy_to_host(self.rows.row(self.rank) + parity_offset, rows)
        no_readers = np.zeros(self.nranks, dtype=bool)
        lacks_header = self._lacks_packets(epoch, rows, no_readers)
        lacks_slot = self._lacks_packets(epoch, rows, readers) & round_active
        peers = [p for p in range(self.nranks) if p != self.rank and has_rank(active_mask, p)]
        for sender in np.flatnonzero(lacks_slot).tolist():
            reads = readers[sender : sender + 1]
            for peer in peers:
                slot = rows[sender : sender + 1].copy()
                self._copy_to_host(
                    self.rows.row(peer) + parity_offset + sender * self._slot_bytes, slot
                )
                if reads[0] and not self._lacks_packets(epoch, slot, reads)[0]:
                    rows[sender] = slot[0]
                    break
                if lacks_header[sender] and not self._lacks_packets(epoch, slot, no_readers[:1])[0]:
                    rows[sender] = slot[0]
                    if not reads[0]:
                        break
        return rows

    def _lacks_packets(self, epoch: int, slots: np.ndarray, readers: np.ndarray) -> np.ndarray:
        """Return which of ``slots`` lack ``epoch``'s header, or for ``readers`` the whole slot."""
        lacking = np.zeros(len(slots), dtype=bool)
        wait_for_slots(
            self._stream.cuda_stream,
            [],
            None,
            slots,
            np.ones(len(slots), dtype=bool),
            readers,
            epoch,
            slots.shape[1],
            lacking,
            self._wake,
            0.0,
        )
        return lacking

    def _copy_to_host(self, src: int, dst: np.ndarray) -> None:
        cuda = self._cuda
        copy = cuda.CUDA_MEMCPY2D()
        copy.srcMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_DEVICE
        copy.srcDevice, copy.srcPitch = src, self._slot_bytes
        copy.dstMemoryType = cuda.CUmemorytype.CU_MEMORYTYPE_HOST
        copy.dstHost, copy.dstPitch = dst.ctypes.data, self._slot_bytes
        copy.WidthInBytes, copy.Height = self._slot_bytes, len(dst)
        _check_cu_result(cuda.cuMemcpy2DAsync(copy, self._stream.cuda_stream))
        self._stream.synchronize()
