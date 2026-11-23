# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The buffered PP transport's protocol (``pp_buffered.py``), on a device simulator.

``BufferedPPComm`` reaches the device only through its ``ops`` object. These tests
give it a simulator: every rank's streams are FIFO queues of operations (copies,
32-bit flag writes, stream-side waits on a flag or an event), and one device
executes the head operation of a randomly chosen ready stream at a time, so each
seed explores another interleaving of the ranks' streams. The hosts enqueue their
whole program first, as the transport never blocks a host. Payloads are stamped per
stage and message and checked where the next stage reads them.
"""

import pickle
import random
import threading
from collections import deque
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.distributed import pp_buffered
from tensorrt_llm._torch.distributed.pp_buffered import (
    BufferedPPComm,
    boundary_bytes_bound,
    buffered_ring_config,
    init_buffered_pp_comm,
    packed_bytes,
)
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

# --------------------------------------------------------------------------- simulator


class _Event:
    def __init__(self):
        self.done = False


class _Stream:
    def __init__(self, name: str):
        self.name = name
        self.ops: deque = deque()


class _Device:
    """Memory, streams and an executor that interleaves the streams at random."""

    def __init__(self):
        self.memory: Dict[int, bytearray] = {}
        self._next_base = 1 << 40
        self.streams: List[_Stream] = []
        self.executed = 0

    def alloc(self, nbytes: int) -> int:
        base = self._next_base
        self._next_base += 1 << 40
        self.memory[base] = bytearray(nbytes)
        return base

    def _locate(self, address: int, nbytes: int):
        for base, buffer in self.memory.items():
            if base <= address and address + nbytes <= base + len(buffer):
                return buffer, address - base
        raise AssertionError(f"access of {nbytes} B at {address:#x} is outside every allocation")

    def read_u32(self, address: int) -> int:
        buffer, offset = self._locate(address, 4)
        return int.from_bytes(buffer[offset : offset + 4], "little")

    def write_u32(self, address: int, value: int) -> None:
        buffer, offset = self._locate(address, 4)
        buffer[offset : offset + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")

    def write(self, address: int, data: bytes) -> None:
        buffer, offset = self._locate(address, len(data))
        buffer[offset : offset + len(data)] = data

    def read(self, address: int, nbytes: int) -> bytes:
        buffer, offset = self._locate(address, nbytes)
        return bytes(buffer[offset : offset + nbytes])

    def new_stream(self, name: str) -> _Stream:
        stream = _Stream(name)
        self.streams.append(stream)
        return stream

    def _ready(self, op) -> bool:
        kind = op[0]
        if kind == "wait_geq":
            _, address, value = op
            return ((self.read_u32(address) - value) & 0xFFFFFFFF) < 0x80000000
        if kind == "wait_event":
            return op[1].done
        return True

    def _execute(self, op) -> None:
        kind = op[0]
        if kind == "wait_geq" or kind == "wait_event":
            return
        if kind == "record":
            op[1].done = True
        elif kind == "write_u32":
            self.write_u32(op[1], op[2])
        elif kind == "fn":
            op[1]()
        else:
            raise AssertionError(kind)

    def step(self, rng: random.Random) -> bool:
        ready = [s for s in self.streams if s.ops and self._ready(s.ops[0])]
        if not ready:
            return False
        stream = rng.choice(ready)
        self._execute(stream.ops.popleft())
        self.executed += 1
        return True

    def run(self, rng: random.Random) -> None:
        while self.step(rng):
            pass
        stuck = {s.name: s.ops[0][0] for s in self.streams if s.ops}
        assert not stuck, f"streams blocked forever: {stuck}"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()


def _fill_from_bytes(tensor: torch.Tensor, data: bytes) -> None:
    flat = tensor.view(-1).view(torch.uint8)
    flat.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))


class _SimOps:
    """``CudaDeviceOps`` on the simulator, for one rank."""

    def __init__(self, device: _Device, rank: int, skip_waits: bool = False):
        self.device = device
        self.rank = rank
        self.compute = device.new_stream(f"rank{rank}.compute")
        self.capturing = False
        self.skip_waits = skip_waits
        self.kept_alive: List[tuple] = []
        self.compute_event_waits = 0

    def current_stream(self):
        return self.compute

    def new_stream(self):
        return self.device.new_stream(f"rank{self.rank}.copy")

    def is_capturing(self) -> bool:
        return self.capturing

    def wait_stream(self, waiter, producer) -> None:
        event = _Event()
        producer.ops.append(("record", event))
        waiter.ops.append(("wait_event", event))

    def record_event(self, stream):
        event = _Event()
        stream.ops.append(("record", event))
        return event

    def wait_event(self, stream, event) -> None:
        if stream is self.compute:
            self.compute_event_waits += 1
        stream.ops.append(("wait_event", event))

    def keep_alive(self, tensor, stream) -> None:
        self.kept_alive.append((tensor, stream))

    def clone(self, tensor, stream):
        out = torch.empty_like(tensor, memory_format=torch.contiguous_format)
        stream.ops.append(("fn", lambda: out.copy_(tensor)))
        return out

    def copy_tensor(self, dst, src, stream) -> None:
        stream.ops.append(("fn", lambda: dst.copy_(src)))

    def alloc_shared(self, nbytes: int):
        base = self.device.alloc(nbytes)
        return base, pickle.dumps(base)

    def open_shared(self, handle: bytes) -> int:
        return pickle.loads(handle)

    def copy_to(self, dst: int, tensor, stream) -> None:
        stream.ops.append(("fn", lambda: self.device.write(dst, _tensor_bytes(tensor))))

    def copy_from(self, tensor, src: int, stream) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        stream.ops.append(("fn", lambda: _fill_from_bytes(tensor, self.device.read(src, nbytes))))

    def write_value(self, address: int, value: int, stream) -> None:
        stream.ops.append(("write_u32", address, value))

    def wait_value_geq(self, address: int, value: int, stream) -> None:
        if self.skip_waits:
            return
        stream.ops.append(("wait_geq", address, value))


class _FakeNccl:
    def __init__(self, mapping):
        self.mapping = mapping
        self.sent: List[torch.Tensor] = []
        self.received: List[torch.Tensor] = []

    def send(self, tensor, dest=None):
        self.sent.append(tensor)

    def recv(self, tensor, src=None):
        self.received.append(tensor)


def _build(
    pp_size: int,
    gpus_per_node: int,
    slots: int,
    slot_bytes: int,
    device: _Device,
    skip_waits_on=(),
    eager_break=None,
):
    """One ``BufferedPPComm`` per rank; their collective setup runs on threads."""
    barrier = threading.Barrier(pp_size)
    rounds: Dict[int, List] = {}
    lock = threading.Lock()
    counters = [0] * pp_size

    def allgather_for(rank):
        def allgather(obj):
            with lock:
                index = counters[rank]
                counters[rank] += 1
                rounds.setdefault(index, [None] * pp_size)[rank] = obj
            barrier.wait()
            result = list(rounds[index])
            barrier.wait()
            return result

        return allgather

    comms: List[Optional[BufferedPPComm]] = [None] * pp_size
    ops = [_SimOps(device, r, skip_waits=r in skip_waits_on) for r in range(pp_size)]
    errors = []

    def make(rank):
        try:
            mapping = Mapping(
                world_size=pp_size, rank=rank, gpus_per_node=gpus_per_node, pp_size=pp_size
            )
            comms[rank] = BufferedPPComm(
                mapping,
                _FakeNccl(mapping),
                slots=slots,
                slot_bytes=slot_bytes,
                ops=ops[rank],
                allgather=allgather_for(rank),
                eager_break=eager_break or (lambda fn: fn),
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=make, args=(r,)) for r in range(pp_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors, errors
    return comms, ops


def _payload(stage: int, message: int, rows: int, hidden: int) -> List[torch.Tensor]:
    """The model's stage boundary in miniature: an FP32 residual and a BF16 attention input."""
    base = float(stage * 1000 + message)
    residual = torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden) + base
    attn_input = (torch.full((rows, hidden), base / 7.0)).to(torch.bfloat16)
    return [residual, attn_input]


def _pipeline_program(comms, ops, messages: int, hidden: int, rows_of, received: Dict):
    """Every stage receives (except the first), stamps, and sends (except the last)."""
    pp_size = len(comms)
    for message in range(messages):
        rows = rows_of(message)
        for stage in range(pp_size):
            comm, op = comms[stage], ops[stage]
            if stage > 0:
                inbox = [
                    torch.empty((rows, hidden), dtype=torch.float32),
                    torch.empty((rows, hidden), dtype=torch.bfloat16),
                ]
                comm.recv_tensors(inbox)

                def check(stage=stage, message=message, inbox=inbox):
                    received.setdefault(stage, []).append((message, [t.clone() for t in inbox]))

                op.compute.ops.append(("fn", check))
            if stage < pp_size - 1:
                comm.send_tensors(_payload(stage, message, rows, hidden))


def _expect_payloads(received, pp_size, messages, hidden, rows_of):
    for stage in range(1, pp_size):
        got = received[stage]
        assert [m for m, _ in got] == list(range(messages)), stage
        for message, tensors in got:
            want = _payload(stage - 1, message, rows_of(message), hidden)
            for a, b in zip(tensors, want):
                assert torch.equal(a, b), (stage, message)


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("slots", [1, 2, 3])
def test_every_message_arrives_intact_and_in_order(seed, slots):
    pp_size, hidden, messages = 4, 16, 9
    rng = random.Random(seed)
    rows_of = lambda m: 1 + (m * 7 + seed) % 5  # noqa: E731 - varying chunk sizes
    slot_bytes = packed_bytes(_payload(0, 0, 5, hidden))
    device = _Device()
    comms, ops = _build(pp_size, 4, slots, slot_bytes, device)
    received: Dict = {}
    _pipeline_program(comms, ops, messages, hidden, rows_of, received)
    device.run(rng)
    _expect_payloads(received, pp_size, messages, hidden, rows_of)
    for stage, comm in enumerate(comms):
        assert comm.send_seq == (messages if stage < pp_size - 1 else 0)
        assert comm.recv_seq == (messages if stage > 0 else 0)
        assert comm.fallback_sends == comm.fallback_recvs == 0
        # One clone per sent tensor, kept alive by the copy stream.
        assert all(stream is comm._copy_stream for _, stream in ops[stage].kept_alive)


def test_the_slot_release_wait_is_load_bearing():
    """Negative control: a sender that skips the release wait overwrites unread slots."""
    pp_size, hidden, messages = 2, 16, 8
    rows_of = lambda m: 4  # noqa: E731
    slot_bytes = packed_bytes(_payload(0, 0, 4, hidden))
    corrupted = 0
    for seed in range(20):
        device = _Device()
        comms, ops = _build(pp_size, 2, 2, slot_bytes, device, skip_waits_on=(0,))
        received: Dict = {}
        # The receiver's program is enqueued after the sender's whole program, so the
        # sender's copy stream can run far ahead.
        for message in range(messages):
            comms[0].send_tensors(_payload(0, message, 4, hidden))
        for message in range(messages):
            inbox = [torch.empty((4, hidden)), torch.empty((4, hidden), dtype=torch.bfloat16)]
            comms[1].recv_tensors(inbox)
            ops[1].compute.ops.append(
                (
                    "fn",
                    lambda m=message, i=inbox: received.setdefault(1, []).append(
                        (m, [t.clone() for t in i])
                    ),
                )
            )
        device.run(random.Random(seed))
        try:
            _expect_payloads(received, pp_size, messages, hidden, rows_of)
        except AssertionError:
            corrupted += 1
    assert corrupted > 0


def test_backlog_bounds_the_clones_kept_alive():
    pp_size, hidden, messages, slots = 2, 8, 7, 2
    slot_bytes = packed_bytes(_payload(0, 0, 3, hidden))
    device = _Device()
    comms, ops = _build(pp_size, 2, slots, slot_bytes, device)
    for message in range(messages):
        comms[0].send_tensors(_payload(0, message, 3, hidden))
    # Message k waits (on the compute stream) for the copy of message k - slots.
    assert ops[0].compute_event_waits == messages - slots
    assert len(comms[0]._backlog) == slots


def test_captured_calls_go_through_nccl_on_both_ends():
    pp_size, hidden = 2, 8
    slot_bytes = packed_bytes(_payload(0, 0, 3, hidden))
    device = _Device()
    comms, ops = _build(pp_size, 2, 2, slot_bytes, device)
    for op in ops:
        op.capturing = True
    comms[0].send_tensors(_payload(0, 0, 3, hidden))
    inbox = [torch.empty((3, hidden)), torch.empty((3, hidden), dtype=torch.bfloat16)]
    comms[1].recv_tensors(inbox)
    assert len(comms[0].fallback.sent) == 2 and len(comms[1].fallback.received) == 2
    assert comms[0].send_seq == comms[1].recv_seq == 0
    assert not any(s.ops for s in device.streams)


def test_oversized_messages_go_through_nccl_on_both_ends():
    pp_size, hidden = 2, 64
    slot_bytes = packed_bytes(_payload(0, 0, 3, hidden))
    device = _Device()
    comms, _ = _build(pp_size, 2, 2, slot_bytes, device)
    big = _payload(0, 0, 4, hidden)
    assert not comms[0].takes_send(big)
    assert not comms[1].takes_recv([torch.empty_like(t) for t in big])
    assert comms[0].takes_send(_payload(0, 0, 3, hidden))
    comms[0].send_tensors(big)
    comms[1].recv_tensors([torch.empty_like(t) for t in big])
    assert comms[0].fallback_sends == comms[1].fallback_recvs == 1
    assert comms[0].send_seq == comms[1].recv_seq == 0


def test_cross_node_edges_use_nccl_and_same_node_edges_the_ring():
    """PP4 over two nodes of two GPUs: only the edge 1 -> 2 crosses nodes."""
    device = _Device()
    comms, _ = _build(4, 2, 2, 4096, device)
    assert [c.send_on_ring for c in comms] == [True, False, True, False]
    assert [c.recv_on_ring for c in comms] == [False, True, False, True]


def test_other_peers_go_through_nccl():
    device = _Device()
    comms, _ = _build(4, 4, 2, 4096, device)
    tensors = [torch.zeros(2, 2)]
    assert not comms[1].takes_send(tensors, dest=3)
    assert comms[1].takes_send(tensors, dest=2)
    assert not comms[2].takes_recv(tensors, src=0)
    assert comms[2].takes_recv(tensors, src=1)


def test_messages_are_eager_breaks_of_a_breakable_graph():
    wrapped = []

    def eager_break(fn):
        wrapped.append(fn.__name__)
        return fn

    device = _Device()
    comms, _ = _build(2, 2, 2, 4096, device, eager_break=eager_break)
    assert sorted(wrapped) == ["recv_tensors", "recv_tensors", "send_tensors", "send_tensors"]
    assert comms[0].send_message.__name__ == "send_tensors"


def test_a_failed_allocation_fails_every_rank():
    device = _Device()
    original = _SimOps.alloc_shared

    def failing(self, nbytes):
        if self.rank == 1:
            raise RuntimeError("out of memory")
        return original(self, nbytes)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_SimOps, "alloc_shared", failing)
        with pytest.raises(AssertionError) as info:
            _build(3, 3, 2, 4096, device)
    assert "could not allocate" in str(info.value)


def test_ring_config():
    assert buffered_ring_config() == (2, 896 << 20)


def test_init_keeps_a_live_communicator_with_the_same_edges(monkeypatch):
    class _Nccl:
        def __init__(self, mapping):
            self.mapping = mapping

    def fake_init(self, mapping, fallback, slots, slot_bytes):
        self._mapping = mapping
        self.fallback = fallback
        self.slots = slots

    monkeypatch.setattr(BufferedPPComm, "__init__", fake_init)
    mapping = Mapping(world_size=4, rank=1, gpus_per_node=4, pp_size=4)
    nccl = _Nccl(mapping)
    # A live stock communicator on the same world becomes the fallback.
    first = init_buffered_pp_comm(nccl, mapping, _Nccl, _Nccl)
    assert isinstance(first, BufferedPPComm) and first.fallback is nccl
    # The same edges on the same world: kept, mapping refreshed on both.
    again = Mapping(world_size=4, rank=1, gpus_per_node=4, pp_size=4)
    assert init_buffered_pp_comm(first, again, _Nccl, _Nccl) is first
    assert first.mapping is again and nccl.mapping is again
    # Different edges on the same workers: refused instead of silently rebinding.
    other = Mapping(world_size=4, rank=1, gpus_per_node=4, tp_size=2, pp_size=2)
    with pytest.raises(RuntimeError, match="different pipeline edges"):
        init_buffered_pp_comm(first, other, _Nccl, _Nccl)
    # Nothing live: a fresh stock communicator is built as the fallback.
    fresh = init_buffered_pp_comm(None, mapping, _Nccl, _Nccl)
    assert isinstance(fresh.fallback, _Nccl) and fresh.fallback is not nccl


# --------------------------------------------------------------------------- dispatch


def test_pp_tensor_ops_keep_the_stock_path_without_the_buffered_transport(monkeypatch):
    from tensorrt_llm._torch.distributed import communicator

    # Even a communicator that has the message methods is driven tensor by
    # tensor unless init_pp_comm installed the buffered transport.
    stock = MagicMock(spec=["send", "recv", "mapping", "send_message", "recv_message"])
    monkeypatch.setattr(communicator, "_pp_comm", stock)
    monkeypatch.setattr(communicator, "_pp_comm_buffered", False)
    tensors = [torch.zeros(2), torch.ones(3)]
    communicator.pp_send_tensors(tensors)
    communicator.pp_recv_tensors(tensors)
    assert [c.args[0] for c in stock.send.call_args_list] == tensors
    assert [c.args[0] for c in stock.recv.call_args_list] == tensors
    assert stock.send_message.call_count == stock.recv_message.call_count == 0


def test_pp_tensor_ops_send_one_message_with_the_buffered_transport(monkeypatch):
    from tensorrt_llm._torch.distributed import communicator

    buffered = MagicMock(spec=["send_message", "recv_message", "send", "recv", "mapping"])
    monkeypatch.setattr(communicator, "_pp_comm", buffered)
    monkeypatch.setattr(communicator, "_pp_comm_buffered", True)
    tensors = [torch.zeros(2), torch.ones(3)]
    communicator.pp_send_tensors(tensors)
    communicator.pp_recv_tensors(tensors)
    assert buffered.send_message.call_count == 1 and buffered.recv_message.call_count == 1
    assert buffered.send.call_count == buffered.recv.call_count == 0


@pytest.mark.parametrize("buffered, pp_size", [(False, 2), (True, 1)])
def test_init_pp_comm_without_the_buffered_request_builds_the_stock_communicator(
    buffered, pp_size, monkeypatch
):
    """An aggregated server (no buffered request) and a one-stage mapping keep the
    stock communicator."""
    from tensorrt_llm._torch.distributed import communicator

    monkeypatch.setattr(communicator, "mpi_disabled", lambda: False)
    monkeypatch.setattr(communicator, "init_helix_cp_comm", lambda mapping: None)
    monkeypatch.setattr(communicator, "_pp_comm", None)

    class _Stock:
        def __init__(self, mapping):
            self.mapping = mapping

    monkeypatch.setattr(communicator, "PPCommNCCL", _Stock)
    monkeypatch.setattr(communicator, "_pp_comm_buffered", False)
    monkeypatch.setattr(
        pp_buffered,
        "init_buffered_pp_comm",
        MagicMock(side_effect=AssertionError("buffered path taken")),
    )
    mapping = Mapping(world_size=pp_size, rank=0, gpus_per_node=2, pp_size=pp_size)
    communicator.init_pp_comm(mapping, buffered=buffered)
    assert type(communicator._pp_comm) is _Stock
    assert communicator.pp_comm_buffered() is False


def test_init_pp_comm_buffered_request_builds_the_buffered_communicator(monkeypatch):
    from tensorrt_llm._torch.distributed import communicator

    monkeypatch.setattr(communicator, "mpi_disabled", lambda: False)
    monkeypatch.setattr(communicator, "init_helix_cp_comm", lambda mapping: None)
    monkeypatch.setattr(communicator, "_pp_comm", None)
    monkeypatch.setattr(communicator, "_pp_comm_buffered", False)
    sentinel = object()
    calls = []

    def fake_init(current, mapping, make_fallback, fallback_type):
        calls.append((current, mapping, make_fallback, fallback_type))
        return sentinel

    monkeypatch.setattr(pp_buffered, "init_buffered_pp_comm", fake_init)
    mapping = Mapping(world_size=2, rank=0, gpus_per_node=2, pp_size=2)
    communicator.init_pp_comm(mapping, buffered=True)
    assert communicator._pp_comm is sentinel
    assert communicator.pp_comm_buffered() is True
    assert calls[0][2] is communicator.PPCommNCCL


# --------------------------------------------------------------------------- further checks


@pytest.mark.parametrize("seed", range(6))
def test_sequence_numbers_wrap_around_32_bits(seed):
    """The flag words are 32-bit and compared cyclically; messages across 2**32 still flow."""
    pp_size, hidden, messages, slots = 3, 8, 7, 2
    start = (1 << 32) - 3
    rows_of = lambda m: 1 + m % 3  # noqa: E731
    slot_bytes = packed_bytes(_payload(0, 0, 3, hidden))
    device = _Device()
    comms, ops = _build(pp_size, 4, slots, slot_bytes, device)
    for comm in comms:
        # As if ``start`` messages had already crossed every edge.
        comm.send_seq = comm.recv_seq = start
        device.write_u32(comm._local_full, start)
        device.write_u32(comm._local_free, start)
    received: Dict = {}
    _pipeline_program(comms, ops, messages, hidden, rows_of, received)
    device.run(random.Random(seed))
    _expect_payloads(received, pp_size, messages, hidden, rows_of)
    assert comms[0].send_seq == start + messages


@pytest.mark.parametrize("seed", range(6))
def test_fallback_messages_interleave_with_ring_messages(seed):
    """Oversized and captured messages mid-stream take NCCL on both ends without shifting
    the ring's sequence numbers or slots."""
    pp_size, hidden, messages, slots = 3, 64, 10, 2
    slot_bytes = packed_bytes(_payload(0, 0, 3, hidden))
    rows_of = lambda m: 5 if m % 4 == 1 else 1 + m % 3  # noqa: E731 - every 4th is oversized
    assert packed_bytes(_payload(0, 0, 5, hidden)) > slot_bytes
    captured = {6}
    device = _Device()
    comms, ops = _build(pp_size, 4, slots, slot_bytes, device)
    received: Dict = {}
    for message in range(messages):
        for op in ops:
            op.capturing = message in captured
        rows = rows_of(message)
        for stage in range(pp_size):
            comm, op = comms[stage], ops[stage]
            if stage > 0:
                inbox = [
                    torch.empty((rows, hidden)),
                    torch.empty((rows, hidden), dtype=torch.bfloat16),
                ]
                comm.recv_tensors(inbox)
                on_ring = not (comm.fallback.received and comm.fallback.received[-1] is inbox[-1])
                if on_ring:

                    def check(stage=stage, message=message, inbox=inbox):
                        received.setdefault(stage, []).append((message, [t.clone() for t in inbox]))

                    op.compute.ops.append(("fn", check))
            if stage < pp_size - 1:
                comm.send_tensors(_payload(stage, message, rows, hidden))
    device.run(random.Random(seed))
    ring = [m for m in range(messages) if rows_of(m) <= 3 and m not in captured]
    for stage in range(1, pp_size):
        assert [m for m, _ in received[stage]] == ring, stage
        for message, tensors in received[stage]:
            want = _payload(stage - 1, message, rows_of(message), hidden)
            assert all(torch.equal(a, b) for a, b in zip(tensors, want)), (stage, message)
    for stage, comm in enumerate(comms):
        fallback = messages - len(ring)
        assert comm.send_seq == (len(ring) if stage < pp_size - 1 else 0)
        assert comm.recv_seq == (len(ring) if stage > 0 else 0)
        assert comm.fallback_sends == (fallback if stage < pp_size - 1 else 0)
        assert comm.fallback_recvs == (fallback if stage > 0 else 0)


def test_ranks_with_different_slot_geometry_are_refused():
    """A sender addresses the successor's slots with its own geometry: refuse a mismatch."""
    device = _Device()
    original = BufferedPPComm.__init__

    def init(self, mapping, fallback, slots=2, slot_bytes=4096, **kwargs):
        # Rank 1 was built with another slot count.
        original(
            self,
            mapping,
            fallback,
            slots=3 if mapping.rank == 1 else slots,
            slot_bytes=slot_bytes,
            **kwargs,
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(BufferedPPComm, "__init__", init)
        with pytest.raises(AssertionError) as info:
            _build(3, 3, 2, 4096, device)
    assert "disagree on the staging ring" in str(info.value)


# --------------------------------------------------------------------------- slot capacity


def test_boundary_bound_is_the_model_boundary_at_the_default_slot():
    # 8192 rows of an FP32 residual + BF16 activations at hidden 8192 fill
    # 384 MiB exactly; one more row does not fit. The default slot holds more.
    assert boundary_bytes_bound(8192, 8192, 2) == 384 << 20
    assert boundary_bytes_bound(8193, 8192, 2) > 384 << 20
    assert pp_buffered.DEFAULT_SLOT_MB << 20 > 384 << 20
    rows = [
        torch.empty(8192, 8192, dtype=torch.float32, device="meta"),
        torch.empty(8192, 8192, dtype=torch.bfloat16, device="meta"),
    ]
    assert boundary_bytes_bound(8192, 8192, 2) == packed_bytes(rows)


@pytest.mark.parametrize("gpus_per_node", [4, 1], ids=["same_node", "cross_node"])
def test_a_slot_smaller_than_the_largest_boundary_warns(gpus_per_node, monkeypatch):
    """With max_num_tokens above what a slot holds the largest messages fall back
    to NCCL; the check warns at init on a rank with an edge on the ring."""
    warnings = []
    monkeypatch.setattr(pp_buffered.logger, "warning", warnings.append)
    hidden, slot_bytes = 64, 64 * 1024
    comms, _ = _build(4, gpus_per_node, 2, slot_bytes, _Device())
    fits = slot_bytes // (hidden * 6)
    for comm in comms:
        comm.check_capacity(fits, hidden, 2)
    for comm in comms:
        if gpus_per_node == 1:
            # No edge on the ring: NCCL carries every message anyway.
            comm.check_capacity(fits + 1, hidden, 2)
            continue
        before = len(warnings)
        comm.check_capacity(fits + 1, hidden, 2)
        assert len(warnings) == before + 1 and "go through NCCL" in warnings[-1]


def test_capacity_check_is_a_no_op_without_the_buffered_transport(monkeypatch):
    from tensorrt_llm._torch.distributed import communicator

    # Without the buffered transport nothing is looked up or called, even on a
    # communicator that has a check_capacity method.
    stock = MagicMock(spec=["send", "recv", "mapping", "check_capacity"])
    monkeypatch.setattr(communicator, "_pp_comm", stock)
    monkeypatch.setattr(communicator, "_pp_comm_buffered", False)
    communicator.check_pp_comm_capacity(1 << 20, 8192, 2)
    stock.check_capacity.assert_not_called()
    buffered = MagicMock(spec=["check_capacity"])
    monkeypatch.setattr(communicator, "_pp_comm", buffered)
    monkeypatch.setattr(communicator, "_pp_comm_buffered", True)
    communicator.check_pp_comm_capacity(16384, 8192, 2)
    buffered.check_capacity.assert_called_once_with(16384, 8192, 2)


# --------------------------------------------------------------------------- owned tensors


def _capture(stage: int, message: int, rows: int, hidden: int) -> torch.Tensor:
    """A fresh tensor the sender hands over (a packed capture carry in miniature)."""
    base = float(stage * 1000 + message)
    return (
        torch.arange(2 * rows * hidden, dtype=torch.float32).reshape(2, rows, hidden) - base
    ).to(torch.bfloat16)


def _counting_clones(ops):
    clones = []
    original = ops.clone

    def clone(tensor, stream):
        clones.append(tensor)
        return original(tensor, stream)

    ops.clone = clone
    return clones


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("slots", [1, 2])
def test_owned_tensors_are_sent_without_a_clone_and_arrive_intact(seed, slots):
    pp_size, hidden, messages, rows = 3, 16, 7, 3
    slot_bytes = packed_bytes([*_payload(0, 0, rows, hidden), _capture(0, 0, rows, hidden)])
    device = _Device()
    comms, ops = _build(pp_size, 4, slots, slot_bytes, device)
    clones = [_counting_clones(op) for op in ops]
    received: Dict = {}
    handed: Dict = {}
    for message in range(messages):
        for stage in range(pp_size):
            if stage > 0:
                inbox = [
                    torch.empty((rows, hidden), dtype=torch.float32),
                    torch.empty((rows, hidden), dtype=torch.bfloat16),
                    torch.empty((2, rows, hidden), dtype=torch.bfloat16),
                ]
                comms[stage].recv_tensors(inbox)
                ops[stage].compute.ops.append(
                    (
                        "fn",
                        lambda s=stage, m=message, i=inbox: received.setdefault(s, []).append(
                            (m, [t.clone() for t in i])
                        ),
                    )
                )
            if stage < pp_size - 1:
                capture = _capture(stage, message, rows, hidden)
                handed.setdefault(stage, []).append(capture)
                comms[stage].send_tensors(
                    [*_payload(stage, message, rows, hidden), capture], owned=(False, False, True)
                )
    device.run(random.Random(seed))
    for stage in range(1, pp_size):
        assert [m for m, _ in received[stage]] == list(range(messages))
        for message, tensors in received[stage]:
            want = [
                *_payload(stage - 1, message, rows, hidden),
                _capture(stage - 1, message, rows, hidden),
            ]
            for got, expected in zip(tensors, want):
                assert torch.equal(got, expected), (stage, message)
    for stage in range(pp_size - 1):
        # The boundary is cloned, the owned capture is not; the copy stream keeps the
        # capture itself alive.
        assert len(clones[stage]) == 2 * messages
        assert not any(c is h for c in clones[stage] for h in handed[stage])
        kept = [tensor for tensor, stream in ops[stage].kept_alive]
        assert all(any(k is h for k in kept) for h in handed[stage])
        assert all(stream is comms[stage]._copy_stream for _, stream in ops[stage].kept_alive)


def test_an_owned_tensor_that_is_not_contiguous_is_cloned():
    hidden, rows = 8, 4
    capture = _capture(0, 0, rows, hidden).transpose(0, 1)
    assert not capture.is_contiguous()
    slot_bytes = packed_bytes([*_payload(0, 0, rows, hidden), capture])
    device = _Device()
    comms, ops = _build(2, 2, 2, slot_bytes, device)
    clones = _counting_clones(ops[0])
    comms[0].send_tensors([*_payload(0, 0, rows, hidden), capture], owned=(False, False, True))
    inbox = [
        torch.empty((rows, hidden)),
        torch.empty((rows, hidden), dtype=torch.bfloat16),
        torch.empty_like(capture, memory_format=torch.contiguous_format),
    ]
    comms[1].recv_tensors(inbox)
    device.run(random.Random(0))
    assert len(clones) == 3 and clones[2] is capture
    assert torch.equal(inbox[2], capture)


def test_ownership_flags_must_name_every_tensor():
    device = _Device()
    comms, _ = _build(2, 2, 2, 4096, device)
    with pytest.raises(ValueError, match="2 ownership flags for a message of 3 tensors"):
        comms[0].send_tensors([torch.zeros(2), torch.zeros(2), torch.zeros(2)], owned=(True, True))
    assert comms[0].send_seq == 0


def test_owned_tensors_of_a_message_that_falls_back_go_through_nccl_as_they_are():
    hidden = 64
    slot_bytes = packed_bytes(_payload(0, 0, 3, hidden))
    device = _Device()
    comms, ops = _build(2, 2, 2, slot_bytes, device)
    clones = _counting_clones(ops[0])
    big = [*_payload(0, 0, 4, hidden)]
    comms[0].send_tensors(big, owned=(True, True))
    assert comms[0].fallback_sends == 1 and comms[0].send_seq == 0
    assert all(a is b for a, b in zip(comms[0].fallback.sent, big))
    assert clones == []


# --------------------------------------------------------------------------- views of reused buffers


def _send_a_view_then_rewrite(device, comms, ops, hidden):
    """The sender sends rows 1-3 of a buffer as a view and then, on its compute stream,
    rewrites the buffer (as the next step's capture write does); returns the rows as sent and
    the receiver's copy (filled when the device runs)."""
    buffer = torch.arange(6 * hidden, dtype=torch.float32).reshape(6, hidden)
    want = buffer[1:4].clone()
    comms[0].send_tensors([buffer.narrow(0, 1, 3)])
    ops[0].compute.ops.append(("fn", lambda: buffer.fill_(-1.0)))
    inbox = torch.empty((3, hidden))
    comms[1].recv_tensors([inbox])
    got: List[torch.Tensor] = []
    ops[1].compute.ops.append(("fn", lambda: got.append(inbox.clone())))
    return want, got


@pytest.mark.parametrize("seed", range(12))
def test_a_sent_view_of_a_buffer_rewritten_after_the_send_arrives_as_sent(seed):
    """A sender may send a view of a buffer it rewrites on its compute stream after the send:
    the clone on the compute stream is the ring's only read of the view."""
    hidden = 8
    device = _Device()
    comms, ops = _build(2, 2, 2, packed_bytes([torch.empty((6, hidden))]), device)
    want, got = _send_a_view_then_rewrite(device, comms, ops, hidden)
    device.run(random.Random(seed))
    assert torch.equal(got[0], want)


def test_the_view_send_check_is_load_bearing():
    """Negative control: without the clone on the compute stream, the copy stream reads the
    view after the rewrite in some interleavings."""
    hidden = 8
    corrupted = 0
    for seed in range(20):
        device = _Device()
        comms, ops = _build(2, 2, 2, packed_bytes([torch.empty((6, hidden))]), device)
        ops[0].clone = lambda tensor, stream: tensor
        want, got = _send_a_view_then_rewrite(device, comms, ops, hidden)
        device.run(random.Random(seed))
        corrupted += not torch.equal(got[0], want)
    assert corrupted > 0


def test_an_oversized_view_goes_to_the_nccl_fallback_as_the_view():
    """A view the ring cannot take is handed to the fallback as it is; the fallback
    (``PPCommNCCL.send``) clones it on the caller's stream before its side stream sends."""
    hidden = 256
    device = _Device()
    comms, _ = _build(2, 2, 2, packed_bytes([torch.empty((1, hidden))]), device)
    buffer = torch.zeros((12, hidden))
    view = buffer.narrow(0, 1, 6)
    assert not comms[0].takes_send([view])
    comms[0].send_tensors([view])
    assert comms[0].fallback_sends == 1
    assert comms[0].fallback.sent[0] is view


def test_the_nccl_send_clones_its_tensor_on_the_callers_stream(monkeypatch):
    """``PPCommNCCL.send`` outside a capture: the clone is made before the side stream waits on
    the caller's stream and sends, so a rewrite of the source after the call cannot reach the
    message."""
    import contextlib
    import types

    from tensorrt_llm._torch.distributed import communicator

    events: List[tuple] = []
    comm = object.__new__(communicator.PPCommNCCL)
    comm.nccl_comm = types.SimpleNamespace(
        send=lambda tensor, dest: events.append(("send", tensor, dest))
    )
    comm.send_stream = MagicMock()
    comm.send_stream.wait_stream.side_effect = lambda stream: events.append(("wait", stream))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "caller")
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    buffer = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    want = buffer[1:4].clone()
    comm.send(buffer.narrow(0, 1, 3), 1)
    buffer.fill_(-1.0)
    assert [event[0] for event in events] == ["wait", "send"]
    assert events[0][1] == "caller"
    sent = events[1][1]
    assert sent.data_ptr() != buffer.data_ptr() + 8 * buffer.element_size()
    assert torch.equal(sent, want)
