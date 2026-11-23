# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Buffered pipeline-parallel point-to-point transport: a copy-engine staging ring.

What it replaces. ``PPCommNCCL`` sends and receives the stage-boundary tensors over
one world NCCL communicator: the send is issued on a side stream and the receive in
the compute stream, and NCCL orders the two in issue order, so a stage's receive of
micro-batch ``j + 1`` waits behind its send of micro-batch ``j``, which completes
only when the next stage posts the matching receive. A stage therefore cannot run
ahead of its successor by even one micro-batch (blocking after service), and a
pre-posted NCCL receive would spin on SMs for the whole compute period.

What runs instead on a disaggregated server at ``pipeline_parallel_size > 1``
(the model engine asks ``init_pp_comm`` for it there; every other deployment
keeps ``PPCommNCCL``). Every rank owns ``slots``
staging slots of ``slot_bytes`` bytes for the edge from its previous stage, plus one
``full`` word, and one ``free`` word for the edge to its next stage; the allocation
is shared with the neighbours by CUDA IPC. For the ``k``-th message on an edge
(one ``pp_send_tensors`` call, all its tensors packed in slot ``(k - 1) % slots``):

* sender: a clone of each tensor on the compute stream (as ``PPCommNCCL`` does:
  the tensor may live in a CUDA graph's pool or a user buffer that the next step
  overwrites; a contiguous tensor the caller hands over with ``owned`` is sent
  as it is), then, on a dedicated copy stream that first waits for the compute
  stream: ``cuStreamWaitValue32(free >= k - slots)`` (the receiver released the
  slot's previous message), one copy-engine peer copy per tensor into the
  receiver's slot, then a fenced ``cuStreamWriteValue32(receiver.full = k)``;
* receiver, on its compute stream: ``cuStreamWaitValue32(full >= k)``, a local copy
  of the slot into the tensors the forward passed in, then a fenced
  ``cuStreamWriteValue32(sender.free = k)``.

The waits are stream-side (the front end, no SM), the copies use the copy engines.
The sender's compute stream never waits for the receiver except through the bounded
backlog: a send waits (stream-side) for the copy of message ``k - slots``, so at most
``slots`` clones are kept alive by the copy stream.

Out of graphs. ``pp_send_tensors`` / ``pp_recv_tensors`` reach this transport
through ``send_message`` / ``recv_message``, which are eager breaks of a breakable
prefill CUDA graph (``eager_on_graph``): the graph covers the stage's layers, the
receive runs before its replay and the send after it, so the sequence numbers
advance on every replay. Outside a breakable graph they are plain calls.

Ownership and progress:

* One writer per slot: the sender of the edge, only after ``free`` released the
  slot's previous message; one reader: the receiver, only after ``full`` published
  the message.
* Publication: the peer copy completes, then the fenced flag write on the same
  stream; acquire: the stream-side wait, then the copy out. Same ordering argument
  as the copy-engine attention-input gather of the TEP target
  (``kernels/context_sequence_parallel/ce_gather.py``).
* Every wait targets a message the peer enqueued earlier in its own stream order, so
  no wait depends on a later one. A peer that never sends leaves the stream waiting
  (as an NCCL receive does); the executor's hang detector covers it.

Fallbacks, decided identically on both ends of an edge so both use the same
transport for every message: an edge whose ranks sit on different nodes, a call made
while the stream is being captured into a (non-breakable) CUDA graph -- the send is
then captured exactly as ``PPCommNCCL`` captures it -- and a message larger than a
slot go through the NCCL communicator. Nothing of the staging ring is ever captured.
Unlike ``PPCommNCCL`` alone, this requires that every stage replays a captured graph
for the same micro-batches (true when all ranks capture the same batch sizes): a
message sent in a graph over NCCL and received eagerly on the ring would never match.

Requirement: the tensors of one ``pp_send_tensors`` / ``pp_recv_tensors`` call
have the same shapes and dtypes on both ends (as for NCCL).
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, List, Optional, Sequence

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

#: Staging slots per edge: two slots let one copy-engine transfer overlap the next
#: micro-batch's compute.
DEFAULT_SLOTS = 2
#: MiB of one slot. An 8192-row FP32 residual plus its BF16 next-layer attention
#: input at hidden 8192 (MuseSparkOpen's stage boundary) is 384 MiB; the slot also
#: holds the DFlash captures a context server's stage boundary carries. A larger
#: message goes through NCCL.
DEFAULT_SLOT_MB = 896
#: Alignment of each tensor inside a slot and of the slot region.
TENSOR_ALIGN = 256
_REGION_ALIGN = 2 << 20
#: Bytes between the ``full`` and ``free`` words (one 128-byte line each).
FLAG_LINE = 128
_U32 = 0xFFFFFFFF


def buffered_ring_config() -> tuple:
    """``(slots, slot_bytes)`` of the staging ring."""
    return DEFAULT_SLOTS, DEFAULT_SLOT_MB << 20


def _align(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def packed_bytes(tensors: Sequence[torch.Tensor]) -> int:
    """Bytes the tensors take in one slot, each at a ``TENSOR_ALIGN`` boundary."""
    return sum(_align(t.numel() * t.element_size(), TENSOR_ALIGN) for t in tensors)


def boundary_bytes_bound(max_rows: int, hidden_size: int, activation_element_size: int) -> int:
    """Upper bound of one stage-boundary message of ``max_rows`` tokens, packed in a slot.

    A stage hands over the residual stream and the next layer's input: at most an
    FP32 residual plus one activation-dtype tensor of ``hidden_size`` per token
    (MuseSparkOpen sends exactly that; models that send both in the activation dtype
    send less).
    """
    rows_by_hidden = int(max_rows) * int(hidden_size)
    return _align(rows_by_hidden * 4, TENSOR_ALIGN) + _align(
        rows_by_hidden * int(activation_element_size), TENSOR_ALIGN
    )


class CudaDeviceOps:
    """The device operations of the transport, on CUDA (driver API through cuda-python)."""

    def __init__(self):
        from cuda.bindings import driver, runtime

        from tensorrt_llm._ipc_utils import _ipc_mem_handle_from_bytes, _ipc_mem_handle_to_bytes

        self._cu = driver
        self._rt = runtime
        self._to_bytes = _ipc_mem_handle_to_bytes
        self._from_bytes = _ipc_mem_handle_from_bytes

    def _check(self, result) -> Any:
        err = result[0] if isinstance(result, tuple) else result
        if isinstance(err, self._rt.cudaError_t):
            if err != self._rt.cudaError_t.cudaSuccess:
                raise RuntimeError(f"PP buffered P2P: CUDA runtime call failed: {err}")
        elif err != self._cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"PP buffered P2P: CUDA driver call failed: {err}")
        if isinstance(result, tuple) and len(result) > 1:
            return result[1]
        return None

    def _stream(self, stream) -> Any:
        return self._cu.CUstream(stream.cuda_stream)

    def current_stream(self):
        return torch.cuda.current_stream()

    def new_stream(self):
        # High priority: the copies are short and should not queue behind
        # compute work issued on other streams.
        return torch.cuda.Stream(priority=-1)

    def is_capturing(self) -> bool:
        return torch.cuda.is_current_stream_capturing()

    def wait_stream(self, waiter, producer) -> None:
        waiter.wait_stream(producer)

    def record_event(self, stream):
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def wait_event(self, stream, event) -> None:
        stream.wait_event(event)

    def keep_alive(self, tensor: torch.Tensor, stream) -> None:
        tensor.record_stream(stream)

    def clone(self, tensor: torch.Tensor, stream) -> torch.Tensor:
        # ``stream`` is the current stream: the clone runs there in order.
        return tensor.clone(memory_format=torch.contiguous_format)

    def copy_tensor(self, dst: torch.Tensor, src: torch.Tensor, stream) -> None:
        dst.copy_(src)

    def alloc_shared(self, nbytes: int) -> tuple:
        pointer = int(self._check(self._rt.cudaMalloc(nbytes)))
        self._check(self._rt.cudaMemset(pointer, 0, nbytes))
        self._check(self._rt.cudaDeviceSynchronize())
        handle = self._check(self._rt.cudaIpcGetMemHandle(pointer))
        return pointer, self._to_bytes(handle)

    def open_shared(self, handle: bytes) -> int:
        return int(
            self._check(
                self._rt.cudaIpcOpenMemHandle(
                    self._from_bytes(handle), self._rt.cudaIpcMemLazyEnablePeerAccess
                )
            )
        )

    def copy_to(self, dst: int, tensor: torch.Tensor, stream) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes:
            self._check(
                self._cu.cuMemcpyAsync(
                    self._cu.CUdeviceptr(dst),
                    self._cu.CUdeviceptr(tensor.data_ptr()),
                    nbytes,
                    self._stream(stream),
                )
            )

    def copy_from(self, tensor: torch.Tensor, src: int, stream) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if nbytes:
            self._check(
                self._cu.cuMemcpyAsync(
                    self._cu.CUdeviceptr(tensor.data_ptr()),
                    self._cu.CUdeviceptr(src),
                    nbytes,
                    self._stream(stream),
                )
            )

    def write_value(self, address: int, value: int, stream) -> None:
        flags = self._cu.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT
        self._check(
            self._cu.cuStreamWriteValue32(
                self._stream(stream), self._cu.CUdeviceptr(address), value & _U32, flags
            )
        )

    def wait_value_geq(self, address: int, value: int, stream) -> None:
        # GEQ compares (int32)(*address - value) >= 0, so the 32-bit sequence
        # numbers may wrap.
        flags = self._cu.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
        self._check(
            self._cu.cuStreamWaitValue32(
                self._stream(stream), self._cu.CUdeviceptr(address), value & _U32, flags
            )
        )


@dataclass
class _EdgeLayout:
    """Where one rank's shared allocation keeps its slots and flag words."""

    region_bytes: int

    def full_word(self, base: int) -> int:
        return base + self.region_bytes

    def free_word(self, base: int) -> int:
        return base + self.region_bytes + FLAG_LINE

    @property
    def total_bytes(self) -> int:
        return self.region_bytes + 2 * FLAG_LINE


class BufferedPPComm:
    """Pipeline-parallel P2P over a copy-engine staging ring, NCCL for what it does not take.

    Args:
        mapping: the global mapping (``prev_pp_rank`` / ``next_pp_rank`` name the edges).
        fallback: the ``PPCommNCCL`` used for captured calls, cross-node edges and
            messages larger than a slot.
        slots: staging slots per edge.
        slot_bytes: bytes per slot.
        ops: device operations (``CudaDeviceOps`` by default; tests pass a simulator).
        allgather: world all-gather of a picklable object (``mpi_allgather`` by default).
        eager_break: wraps a callable into an eager break of a breakable CUDA graph
            (``eager_on_graph`` by default).
    """

    def __init__(
        self,
        mapping: Mapping,
        fallback,
        slots: int = DEFAULT_SLOTS,
        slot_bytes: int = DEFAULT_SLOT_MB << 20,
        ops=None,
        allgather: Optional[Callable[[Any], List[Any]]] = None,
        eager_break: Optional[Callable[[Callable], Callable]] = None,
    ):
        if slots < 1 or slot_bytes < TENSOR_ALIGN:
            raise ValueError(f"PP buffered P2P: invalid ring of {slots} slots x {slot_bytes} B")
        self.fallback = fallback
        self.slots = int(slots)
        self.slot_bytes = _align(int(slot_bytes), TENSOR_ALIGN)
        self._ops = ops if ops is not None else CudaDeviceOps()
        if allgather is None:
            from tensorrt_llm._utils import mpi_allgather

            allgather = mpi_allgather
        self._mapping = mapping
        self.send_seq = 0
        """Messages sent on the staging ring to the next stage."""
        self.recv_seq = 0
        """Messages received on the staging ring from the previous stage."""
        self.fallback_sends = 0
        self.fallback_recvs = 0
        self._backlog: Deque[Any] = deque()
        self._warned_oversize = False
        self._copy_stream = None
        self._setup(mapping, allgather)
        if eager_break is None:
            from ..pyexecutor.breakable_cuda_graph import eager_on_graph

            eager_break = eager_on_graph
        self.send_message = eager_break(self.send_tensors)
        """``send_tensors`` as an eager break of a breakable CUDA graph."""
        self.recv_message = eager_break(self.recv_tensors)
        """``recv_tensors`` as an eager break of a breakable CUDA graph."""

    # ------------------------------------------------------------------ setup

    @property
    def mapping(self) -> Mapping:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: Mapping) -> None:
        self._mapping = mapping
        self.fallback.mapping = mapping

    def edge_ranks(self, mapping: Optional[Mapping] = None) -> tuple:
        """``(prev, next)`` global ranks of this rank's PP edges (None at the ends)."""
        mapping = self._mapping if mapping is None else mapping
        prev_rank = None if mapping.is_first_pp_rank() else mapping.prev_pp_rank()
        next_rank = None if mapping.is_last_pp_rank() else mapping.next_pp_rank()
        return prev_rank, next_rank

    @staticmethod
    def _same_node(mapping: Mapping, peer: Optional[int]) -> bool:
        return peer is not None and mapping.get_node_rank(peer) == mapping.node_rank

    def _setup(self, mapping: Mapping, allgather) -> None:
        prev_rank, next_rank = self.edge_ranks(mapping)
        self.send_on_ring = self._same_node(mapping, next_rank)
        """Whether the edge to the next stage uses the staging ring."""
        self.recv_on_ring = self._same_node(mapping, prev_rank)
        """Whether the edge from the previous stage uses the staging ring."""
        region = _align(self.slots * self.slot_bytes, _REGION_ALIGN) if self.recv_on_ring else 0
        layout = _EdgeLayout(region)
        error = None
        base, handle = 0, b""
        try:
            base, handle = self._ops.alloc_shared(layout.total_bytes)
        except RuntimeError as exc:
            error = exc
        table = allgather(
            (mapping.rank, handle, region, error is None, self.slots, self.slot_bytes)
        )
        by_rank = {entry[0]: entry for entry in table}
        if not all(entry[3] for entry in table):
            raise RuntimeError(
                "PP buffered P2P: a rank could not allocate its staging ring"
                + (f": {error!r}" if error is not None else "")
            )
        # A sender addresses its successor's slots with its own slot geometry, so every
        # rank must use the same one.
        geometries = {(entry[4], entry[5]) for entry in table}
        if len(geometries) != 1:
            raise RuntimeError(
                "PP buffered P2P: the ranks disagree on the staging ring "
                f"(slots, slot bytes): {sorted(geometries)}."
            )
        self._local_base = base
        self._local_full = layout.full_word(base)
        self._local_free = layout.free_word(base)
        self._local_slots = base
        self._peer_slots = self._peer_full = self._prev_free = None
        error = None
        try:
            if self.send_on_ring:
                _, peer_handle, peer_region, *_ = by_rank[next_rank]
                peer_base = self._ops.open_shared(peer_handle)
                self._peer_slots = peer_base
                self._peer_full = _EdgeLayout(peer_region).full_word(peer_base)
            if self.recv_on_ring:
                _, peer_handle, peer_region, *_ = by_rank[prev_rank]
                peer_base = self._ops.open_shared(peer_handle)
                self._prev_free = _EdgeLayout(peer_region).free_word(peer_base)
        except RuntimeError as exc:
            error = exc
        if not all(allgather(error is None)):
            raise RuntimeError(
                "PP buffered P2P: a rank could not map its neighbours' staging "
                "rings" + (f": {error!r}" if error is not None else "")
            )
        if self.send_on_ring:
            self._copy_stream = self._ops.new_stream()

        def _edge(on_ring: bool, peer: Optional[int]) -> str:
            return "staging ring" if on_ring else "NCCL" if peer is not None else "none"

        logger.info(
            f"PP buffered P2P on rank {mapping.rank}: {self.slots} slots x "
            f"{self.slot_bytes >> 20} MiB; from the previous stage: "
            f"{_edge(self.recv_on_ring, prev_rank)}, to the next stage: "
            f"{_edge(self.send_on_ring, next_rank)}"
        )

    # --------------------------------------------------------------- protocol

    def _slot_offset(self, seq: int) -> int:
        return ((seq - 1) % self.slots) * self.slot_bytes

    def check_capacity(self, max_rows: int, hidden_size: int, activation_element_size: int) -> None:
        """Warn when the slots cannot hold the largest stage boundary.

        Called once the model is known. A message larger than a slot goes
        through NCCL (both ends decide alike), so with ``max_num_tokens`` above
        what a slot holds the largest steps use the stock transport.
        """
        if not (self.send_on_ring or self.recv_on_ring):
            return
        need = boundary_bytes_bound(max_rows, hidden_size, activation_element_size)
        if need <= self.slot_bytes:
            return
        need_mb = -(-need // (1 << 20))
        logger.warning(
            f"PP buffered P2P: a stage boundary of up to {max_rows} tokens x hidden "
            f"{hidden_size} (FP32 residual + {activation_element_size}-byte activations) "
            f"takes up to {need_mb} MiB, more than the {self.slot_bytes >> 20} MiB slot; "
            "such messages go through NCCL."
        )

    def _fits(self, tensors: Sequence[torch.Tensor]) -> bool:
        if packed_bytes(tensors) <= self.slot_bytes:
            return True
        if not self._warned_oversize:
            self._warned_oversize = True
            logger.warning(
                f"PP buffered P2P: a {packed_bytes(tensors)} B stage boundary exceeds the "
                f"{self.slot_bytes} B slot; such messages go through NCCL."
            )
        return False

    def takes_send(self, tensors: Sequence[torch.Tensor], dest: Optional[int] = None) -> bool:
        """Whether a send of ``tensors`` to ``dest`` uses the staging ring (else NCCL)."""
        _, next_rank = self.edge_ranks()
        return (
            self.send_on_ring
            and dest in (None, next_rank)
            and not self._ops.is_capturing()
            and self._fits(tensors)
        )

    def takes_recv(self, tensors: Sequence[torch.Tensor], src: Optional[int] = None) -> bool:
        """Whether a receive of ``tensors`` from ``src`` uses the staging ring (else NCCL)."""
        prev_rank, _ = self.edge_ranks()
        return (
            self.recv_on_ring
            and src in (None, prev_rank)
            and not self._ops.is_capturing()
            and self._fits(tensors)
        )

    def send_tensors(
        self,
        tensors: Sequence[torch.Tensor],
        dest: Optional[int] = None,
        owned: Optional[Sequence[bool]] = None,
    ) -> None:
        """Send ``tensors`` (one message) to the next stage.

        Args:
            tensors: the stage-boundary tensors, any shapes and dtypes, as the
                receiver's ``recv_tensors`` call lists them.
            dest: the destination global rank (default: the next PP rank).
            owned: per tensor, whether the caller hands it over: a tensor that nothing
                writes after this call (a fresh allocation of the caching allocator,
                not a graph pool or a buffer the next step reuses). An owned
                contiguous tensor is copied to the peer as it is, without the
                compute-stream clone; the copy stream keeps it alive until its copy
                is done, as it keeps a clone. None (the default) clones every tensor.

        Raises:
            ValueError: when ``owned`` does not name every tensor of the message.
        """
        if owned is not None and len(owned) != len(tensors):
            raise ValueError(
                f"PP buffered P2P: {len(owned)} ownership flags for a message of "
                f"{len(tensors)} tensors"
            )
        if not self.takes_send(tensors, dest):
            self.fallback_sends += 1
            for tensor in tensors:
                self.fallback.send(tensor, dest)
            return
        ops = self._ops
        self.send_seq += 1
        seq = self.send_seq
        compute = ops.current_stream()
        copy = self._copy_stream
        if len(self._backlog) >= self.slots:
            # Bound the clones kept alive by pending copies to ``slots``.
            ops.wait_event(compute, self._backlog.popleft())
        handed = (False,) * len(tensors) if owned is None else owned
        clones = [
            tensor if keep and tensor.is_contiguous() else ops.clone(tensor, compute)
            for tensor, keep in zip(tensors, handed)
        ]
        ops.wait_stream(copy, compute)
        if seq > self.slots:
            ops.wait_value_geq(self._local_free, seq - self.slots, copy)
        offset = self._peer_slots + self._slot_offset(seq)
        for clone in clones:
            ops.copy_to(offset, clone, copy)
            ops.keep_alive(clone, copy)
            offset += _align(clone.numel() * clone.element_size(), TENSOR_ALIGN)
        ops.write_value(self._peer_full, seq, copy)
        self._backlog.append(ops.record_event(copy))

    def recv_tensors(self, tensors: Sequence[torch.Tensor], src: Optional[int] = None) -> None:
        """Receive one message from the previous stage into ``tensors`` (filled in place).

        Args:
            tensors: preallocated tensors with the shapes and dtypes the sender lists.
            src: the source global rank (default: the previous PP rank).
        """
        if not self.takes_recv(tensors, src):
            self.fallback_recvs += 1
            for tensor in tensors:
                self.fallback.recv(tensor, src)
            return
        ops = self._ops
        self.recv_seq += 1
        seq = self.recv_seq
        compute = ops.current_stream()
        ops.wait_value_geq(self._local_full, seq, compute)
        offset = self._local_slots + self._slot_offset(seq)
        for tensor in tensors:
            if tensor.is_contiguous():
                ops.copy_from(tensor, offset, compute)
            else:
                staged = torch.empty_like(tensor, memory_format=torch.contiguous_format)
                ops.copy_from(staged, offset, compute)
                ops.copy_tensor(tensor, staged, compute)
            offset += _align(tensor.numel() * tensor.element_size(), TENSOR_ALIGN)
        ops.write_value(self._prev_free, seq, compute)

    def send(self, tensor: torch.Tensor, dest: Optional[int] = None) -> None:
        """``PPCommNCCL.send`` for one tensor: a one-tensor message."""
        self.send_tensors([tensor], dest)

    def recv(self, tensor: torch.Tensor, src: Optional[int] = None) -> None:
        """``PPCommNCCL.recv`` for one tensor: a one-tensor message."""
        self.recv_tensors([tensor], src)


def init_buffered_pp_comm(
    current, mapping: Mapping, make_fallback: Callable[[Mapping], Any], fallback_type: type
):
    """Build (or keep) the buffered PP communicator for ``mapping``.

    Called by ``init_pp_comm`` on every rank when it is asked for the buffered
    transport. A live
    buffered communicator on the same world size and the same PP edges is kept (its
    mapping refreshed) so that a second engine on reused workers does not allocate
    or rebind anything; a live ``PPCommNCCL`` becomes the fallback of a new one.

    Args:
        current: the module's current PP communicator (None, ``PPCommNCCL`` or this class).
        mapping: the global mapping.
        make_fallback: builds a ``PPCommNCCL`` for ``mapping``.
        fallback_type: the ``PPCommNCCL`` class.
    """
    if isinstance(current, BufferedPPComm):
        if (
            current.mapping.world_size == mapping.world_size
            and current.edge_ranks() == current.edge_ranks(mapping)
        ):
            current.mapping = mapping
            return current
        raise RuntimeError(
            "PP buffered P2P: a live buffered communicator cannot be rebound to a mapping "
            "with different pipeline edges on the same workers."
        )
    if isinstance(current, fallback_type) and current.mapping.world_size == mapping.world_size:
        current.mapping = mapping
        fallback = current
    else:
        fallback = make_fallback(mapping)
    slots, slot_bytes = buffered_ring_config()
    return BufferedPPComm(mapping, fallback, slots=slots, slot_bytes=slot_bytes)
