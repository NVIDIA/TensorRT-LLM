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
"""CPU tests of the dense DWDP FABRIC row store over a recording fake of the CUDA VMM calls.

The layouts come from toy decoder layers through ``layout.collect_dense_layout``; the big layers
have rows of about 1.25 MiB (``N = 4``), so ``STORE_CHUNK_MB`` splits them.  The store maps its
chunks read-write and zero-filled, writes this rank's rows, exports one handle per chunk, imports
and maps every peer's handles read-only in rank order, and releases peer mappings, imported
handles, local mappings and local handles in that order, also after a failure at any step of
``allocate`` or ``import_peers``, which then does not run again.
"""

import ctypes
import math
import re
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modules.dwdp import transport, vmm
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.layout import collect_dense_layout, write_blob
from tensorrt_llm._torch.modules.dwdp.dense.store import FabricShardStore
from tensorrt_llm._torch.modules.dwdp.dense.types import ROW_ALIGN, DenseLayout

pytestmark = pytest.mark.cpu_only

FABRIC = vmm.cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
PROT_READ = vmm.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
PROT_READWRITE = vmm.cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
GRANULARITY = 2 << 20
GARBAGE = 0xEE  # Fake device memory before the store zero-fills it.

#: Four layers whose rows take about 1.25 MiB each with N = 4 (A 3 MiB + 5 * l bytes, D 2 MiB).
BIG_LAYERS = {
    layer: ([(torch.uint8, ((3 << 20) + 5 * layer,))], [(torch.bfloat16, (1 << 20,))])
    for layer in range(1, 5)
}


class ToyLayer(nn.Module):
    """A decoder layer whose ``attn`` and ``moe`` roots hold one random Parameter per spec."""

    def __init__(self, spec: tuple[list, list], generator: torch.Generator) -> None:
        super().__init__()
        for segment, tensors in zip(("attn", "moe"), spec):
            root = nn.Module()
            for index, (dtype, shape) in enumerate(tensors):
                nbytes = math.prod(shape) * dtype.itemsize
                data = torch.randint(0, 256, (nbytes,), generator=generator, dtype=torch.uint8)
                weight = nn.Parameter(data.view(dtype).reshape(shape), requires_grad=False)
                root.register_parameter(f"w{index}", weight)
            setattr(self, segment, root)

    def dense_dwdp_roots(self, segment: str) -> list[tuple[str, nn.Module]]:
        return [(segment, getattr(self, segment))]


def collect(num_ranks: int, chunk_mb: int) -> DenseLayout:
    """The layout of the ``BIG_LAYERS`` toy layers (seeded, so every call gives equal bytes)."""
    generator = torch.Generator().manual_seed(1234)
    layers = [(index, ToyLayer(BIG_LAYERS[index], generator)) for index in sorted(BIG_LAYERS)]
    config = DenseConfig(enabled=True, min_bytes=64, store_chunk_mb=chunk_mb)
    return collect_dense_layout(
        layers,
        config,
        num_ranks=num_ranks,
        device=torch.device("cpu"),
        attr_allow=frozenset(),
        exclude=frozenset(),
    )


def host_bytes(address: int, nbytes: int) -> torch.Tensor:
    """A ``uint8`` tensor that aliases ``nbytes`` of host memory at ``address``."""
    return torch.frombuffer((ctypes.c_ubyte * nbytes).from_address(address), dtype=torch.uint8)


class FakeVmm:
    """The VMM, export/import and synchronize calls of the FABRIC store, over host memory.

    Every call is recorded as ``(op, value)``; ``cuMemSetAccess`` is faked at the driver binding,
    so the ``access_rw`` / ``access_ro`` records are the access flags the store asks for.
    Reserved address ranges are host buffers filled with garbage, so the store's views alias them.
    Like the driver, the fake knows the size of every handle (created, or imported from a payload
    ``export`` or ``peer_payloads`` registered) and checks that handles and reservations are whole
    granules, a mapping covers one whole handle and exactly its reservation, and access, unmap
    and free name a whole reservation.  ``fail=(op, n)`` makes the ``n``-th call of ``op`` raise.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, fail: tuple[str, int] | None = None
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.fail = fail
        self.counts: Counter = Counter()
        self.memory: dict[int, torch.Tensor] = {}
        self.owner: dict[int, int] = {}  # va -> the handle mapped (or about to be) there
        self.imported: set[int] = set()
        self.handle_sizes: dict[int, int] = {}
        self.payload_sizes: dict[bytes, int] = {}
        self.va_sizes: dict[int, int] = {}
        self._next_handle = 1000
        self._last_handle = None
        patches = {
            (vmm, "peer_handle_type"): lambda: FABRIC,
            (vmm, "get_allocation_granularity"): lambda device_id: GRANULARITY,
            (vmm, "create_fabric_handle"): self.create,
            (vmm, "reserve_va"): self.reserve,
            (vmm, "map_handle"): self.map,
            (vmm.cuda, "cuMemSetAccess"): self.set_access,
            (vmm, "tensor_from_ptr"): self.view,
            (vmm, "unmap_va"): self.unmap,
            (vmm, "free_va"): self.free,
            (vmm, "release_handle"): lambda handle: self._record("release", handle),
            (transport, "_export_handle"): self.export,
            (transport, "_import_handle"): self.import_handle,
            (torch.cuda, "synchronize"): lambda device=None: self._record("sync", device),
            (torch.cuda, "current_stream"): self.current_stream,
        }
        for (owner, name), value in patches.items():
            monkeypatch.setattr(owner, name, value)

    def _record(self, op: str, value: object) -> None:
        self.counts[op] += 1
        if self.fail == (op, self.counts[op]):
            raise RuntimeError(f"injected failure of {op} #{self.counts[op]}")
        self.calls.append((op, value))

    def _new_handle(self, size: int) -> int:
        self._next_handle += 1
        self.handle_sizes[self._next_handle] = size
        self._last_handle = self._next_handle
        return self._next_handle

    def _check_reservation(self, what: str, va: int, size: int) -> None:
        assert self.va_sizes.get(va) == size, f"{what} of {size} bytes at {va:#x}"

    def create(self, size: int, device_id: int) -> int:
        assert size > 0 and size % GRANULARITY == 0, f"cuMemCreate of {size} bytes"
        self._record("create", (self._next_handle + 1, size))
        return self._new_handle(size)

    def import_handle(self, payload: bytes, handle_type: object) -> int:
        assert handle_type == FABRIC and payload in self.payload_sizes, payload
        self._record("import", (self._next_handle + 1, payload))
        self.imported.add(self._next_handle + 1)
        return self._new_handle(self.payload_sizes[payload])

    def export(self, handle: int, handle_type: object) -> bytes:
        assert handle_type == FABRIC
        self._record("export", handle)
        payload = f"fabric-handle-{handle}".encode()
        self.payload_sizes[payload] = self.handle_sizes[handle]
        return payload

    def reserve(self, size: int, alignment: int) -> int:
        assert alignment == GRANULARITY and size > 0 and size % GRANULARITY == 0, size
        self._record("reserve", size)
        buffer = torch.full((size + 2 * ROW_ALIGN,), GARBAGE, dtype=torch.uint8)
        va = buffer.data_ptr() + (-buffer.data_ptr()) % ROW_ALIGN
        self.memory[va] = buffer
        self.va_sizes[va] = size
        self.owner[va] = self._last_handle
        self.calls[-1] = ("reserve", (va, size))
        return va

    def map(self, va: int, size: int, handle: int, offset: int = 0) -> None:
        assert self.owner[va] == handle and offset == 0
        self._check_reservation("cuMemMap", va, size)
        assert size == self.handle_sizes[handle], f"cuMemMap of {size} bytes of handle {handle}"
        self._record("map", (va, handle))

    def set_access(self, va: int, size: int, descriptors: object, count: int) -> tuple:
        """``cuMemSetAccess``: records ``access_rw`` or ``access_ro`` by the requested flags."""
        assert count == len(descriptors) == 1 and descriptors[0].location.id == 0
        self._check_reservation("cuMemSetAccess", va, size)
        flags = int(descriptors[0].flags)
        op = {int(PROT_READWRITE): "access_rw", int(PROT_READ): "access_ro"}.get(flags)
        assert op is not None, f"unexpected access flags {flags}"
        self._record(op, va)
        return (vmm.cuda.CUresult.CUDA_SUCCESS,)

    def view(
        self, ptr: int, shape: tuple[int, ...], dtype: torch.dtype, device_id: int
    ) -> torch.Tensor:
        self._record("view", ptr)
        return host_bytes(ptr, math.prod(shape) * dtype.itemsize).view(dtype).reshape(shape)

    def unmap(self, va: int, size: int) -> None:
        self._check_reservation("cuMemUnmap", va, size)
        self._record("unmap", va)

    def free(self, va: int, size: int) -> None:
        self._check_reservation("cuMemAddressFree", va, size)
        self._record("free", va)
        del self.memory[va]
        del self.va_sizes[va]

    def current_stream(self, device: object = None) -> SimpleNamespace:
        return SimpleNamespace(synchronize=lambda: self._record("stream_sync", device))

    def ops(self, start: int = 0) -> list[str]:
        return [op for op, _ in self.calls[start:]]

    def values(self, op: str, start: int = 0, stop: int | None = None) -> list:
        return [value for name, value in self.calls[start:stop] if name == op]

    def is_peer(self, value: int) -> bool:
        """Whether a recorded handle or address belongs to an imported (peer) handle."""
        return self.owner.get(value, value) in self.imported


def peer_payloads(
    fake: FakeVmm, store: FabricShardStore, layout: DenseLayout, rank: int
) -> list[list[bytes]]:
    """Collective #2 as this rank sees it: its own export plus a distinct export per peer."""
    own = store.export()
    payloads = []
    for peer in range(layout.num_ranks):
        blobs = [f"peer-{peer}-chunk-{chunk.index}".encode() for chunk in layout.chunks]
        for blob, own_blob in zip(blobs, own):
            fake.payload_sizes[blob] = fake.payload_sizes[own_blob]
        payloads.append(own if peer == rank else blobs)
    return payloads


def check_release(fake: FakeVmm, start: int) -> None:
    """The calls since ``start`` undo every handle and mapping created before it, once each.

    Order: a device synchronize first; then the peer mappings (unmap, free) and imported handles,
    then the local mappings and handles; a mapping is unmapped before its VA is freed and both
    happen before its handle is released.
    """
    created = [handle for handle, _ in fake.values("create", 0, start)]
    imported = [handle for handle, _ in fake.values("import", 0, start)]
    reserved = [va for va, _ in fake.values("reserve", 0, start)]
    mapped = [va for va, _ in fake.values("map", 0, start)]
    tail = fake.calls[start:]
    if not (created or imported):
        assert tail == []
        return
    assert tail[0][0] == "sync"
    released = fake.values("release", start)
    assert sorted(released) == sorted(created + imported) and len(set(released)) == len(released)
    assert sorted(fake.values("free", start)) == sorted(reserved)
    assert sorted(fake.values("unmap", start)) == sorted(mapped)
    position = {(op, value): index for index, (op, value) in enumerate(tail)}
    for va in reserved:
        assert position[("free", va)] < position[("release", fake.owner[va])]
        if va in mapped:
            assert position[("unmap", va)] < position[("free", va)]
    peer_steps = [i for i, (_, value) in enumerate(tail[1:]) if fake.is_peer(value)]
    local_steps = [i for i, (_, value) in enumerate(tail[1:]) if not fake.is_peer(value)]
    if peer_steps and local_steps:
        assert max(peer_steps) < min(local_steps), "peer resources must go before local ones"
    peer_frees = [position[("free", va)] for va in reserved if fake.is_peer(va)]
    peer_releases = [position[("release", handle)] for handle in imported]
    if peer_frees and peer_releases:
        assert max(peer_frees) < min(peer_releases), "peer VA ranges go before imported handles"


@pytest.mark.parametrize("chunk_mb", [0, 1], ids=["one-chunk", "chunked"])
def test_fabric_store_maps_writes_exports_and_imports(
    monkeypatch: pytest.MonkeyPatch, chunk_mb: int
) -> None:
    """One handle per chunk mapped read-write and zeroed, this rank's rows, peers read-only."""
    num_ranks, rank = 4, 2
    layout = collect(num_ranks, chunk_mb)
    num_chunks = len(layout.chunks)
    fake = FakeVmm(monkeypatch)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))

    store.allocate()
    per_chunk = ["create", "reserve", "map", "access_rw", "view"]
    assert fake.ops() == per_chunk * num_chunks + ["stream_sync"]
    assert [size for _, size in fake.values("create")] == [c.handle_bytes for c in layout.chunks]
    local_vas = [va for va, _ in fake.values("reserve")]
    for chunk, va in zip(layout.chunks, local_vas):
        assert not host_bytes(va, chunk.handle_bytes).any(), f"chunk {chunk.index} not zeroed"

    store.write_local_rows()
    for chunk, va in zip(layout.chunks, local_vas):
        written = torch.zeros(chunk.handle_bytes, dtype=torch.uint8)
        for layer in chunk.layers:
            for segment in layout.layer_segments(layer):
                width = segment.row_bytes
                blob = torch.empty(num_ranks * width, dtype=torch.uint8)
                write_blob(segment, blob, num_ranks=num_ranks)
                row = blob[rank * width : (rank + 1) * width]
                written[segment.chunk_offset : segment.chunk_offset + width] = row
        assert torch.equal(host_bytes(va, chunk.handle_bytes), written), f"chunk {chunk.index}"

    mark = len(fake.calls)
    payloads = peer_payloads(fake, store, layout, rank)
    assert payloads[rank] == [f"fabric-handle-{h}".encode() for h, _ in fake.values("create")]
    assert store.export() == payloads[rank]  # Exported once, returned again.
    assert fake.ops(mark) == ["export"] * num_chunks

    mark = len(fake.calls)
    store.import_peers(payloads)
    imports = [payload for _, payload in fake.values("import", mark)]
    assert imports == [blob for peer in (0, 1, 3) for blob in payloads[peer]]
    assert fake.ops(mark) == ["import", "reserve", "map", "access_ro"] * (3 * num_chunks)
    peer_sizes = [size for _, size in fake.values("reserve", mark)]
    assert peer_sizes == [chunk.handle_bytes for chunk in layout.chunks] * 3
    peer_vas = iter(va for va, _ in fake.values("reserve", mark))
    expected_bases = tuple(
        tuple(local_vas) if peer == rank else tuple(next(peer_vas) for _ in layout.chunks)
        for peer in range(num_ranks)
    )
    assert store.chunk_bases() == expected_bases

    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    mark = len(fake.calls)
    store.release()
    assert fake.calls[mark:] == [] and fake.memory == {}


#: ``(op, n)`` failure points of ``allocate`` and ``import_peers`` on rank 1 of 3 with four chunks:
#: allocate makes four creates, reserves, maps, read-write accesses and views; import_peers makes
#: eight imports, reserves (calls 5-12 overall), maps and read-only accesses.
FAILURE_POINTS = [
    ("reserve", 2),
    ("map", 3),
    ("view", 4),
    ("import", 6),
    ("reserve", 11),
    ("access_ro", 7),
]


@pytest.mark.parametrize("fail", FAILURE_POINTS, ids=[f"{op}-{n}" for op, n in FAILURE_POINTS])
def test_fabric_release_after_a_partial_allocate_or_import(
    monkeypatch: pytest.MonkeyPatch, fail: tuple[str, int]
) -> None:
    """A step that fails part way does not run again and leaves what it created to ``release``."""
    num_ranks, rank = 3, 1
    layout = collect(num_ranks, chunk_mb=1)
    assert len(layout.chunks) == 4
    fake = FakeVmm(monkeypatch, fail=fail)
    store = FabricShardStore(layout, rank, torch.device("cuda", 0))
    failed = "allocate"
    with pytest.raises(RuntimeError, match="injected failure"):
        store.allocate()
        failed = "write_local_rows"
        store.write_local_rows()
        failed = "export"
        payloads = peer_payloads(fake, store, layout, rank)
        failed = "import_peers"
        store.import_peers(payloads)
    with pytest.raises(RuntimeError):
        store.chunk_bases()
    if failed == "allocate":
        with pytest.raises(RuntimeError, match="allocate already ran"):
            store.allocate()
        for step in (store.write_local_rows, store.export):
            with pytest.raises(RuntimeError, match="before a successful allocate"):
                step()
    elif failed == "import_peers":
        with pytest.raises(RuntimeError, match="import_peers already ran"):
            store.import_peers(payloads)
    mark = len(fake.calls)
    store.release()
    check_release(fake, mark)
    assert fake.memory == {}
    mark = len(fake.calls)
    store.release()
    assert fake.calls[mark:] == []


def test_a_refused_store_handle_is_named_with_its_chunk_size_and_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch keeps the store in one handle: the error states that handle, and no setting."""
    layout = collect(3, chunk_mb=0)
    assert len(layout.chunks) == 1
    FakeVmm(monkeypatch, fail=("create", 1))
    store = FabricShardStore(layout, 1, torch.device("cuda", 0))
    with pytest.raises(RuntimeError) as raised:
        store.allocate()
    message = str(raised.value)
    assert re.fullmatch(
        r"cuMemCreate of the FABRIC handle of store chunk 0 of 1 \(\d+ bytes, layers 1\.\.4\) on "
        r"cuda:0 failed: injected failure of create #1",
        message,
    ), message
