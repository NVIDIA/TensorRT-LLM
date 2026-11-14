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
"""An SM-free transport for TorchDist's object collectives.

TorchDist runs the executor's per-step host collectives (votes) on a side
stream (``TorchDist._cuda_obj_pg``). An NCCL kernel there needs SMs, and a
persistent kernel that holds every SM while it waits for its peer ranks --
MegaMoE under the overlap scheduler -- then deadlocks against it: on one rank
the vote kernel got SMs first and the MegaMoE grid cannot become resident, on
the peer the MegaMoE grid holds every SM and the vote kernel never starts.

This transport moves no byte with an SM. Every vote goes from pinned host
memory straight into the peers' copies of a symmetric buffer (torch symmetric
memory, the process's default backend -- the one MegaMoE maps its own buffers
with), and the ranks synchronize through 64-bit sequence words written the same
way and awaited with stream memory operations. Host-to-device and
device-to-host copies run on copy engines and stream memory operations on the
GPU's front end. No step is a same-GPU copy: some GB200 nodes run those on SMs
even when asked to prefer copy engines
(``cudaMemcpyFlagPreferOverlapWithCompute``), which is why NCCL's own zero-CTA
all-gather -- it copies its local chunk that way -- still needs SMs there.
``verify_sm_free`` proves the transport while a kernel holds every SM before
TorchDist switches to it.

Round protocol (sequence number ``s``; every rank runs the same rounds in the
same order):

1. the round's writers put their bytes into parity ``s % 2`` of every rank's
   buffer (one multicast copy, or one copy per peer);
2. every rank puts ``s`` into its sequence word on every rank;
3. every rank's stream waits until every peer's sequence word reaches ``s``;
4. the readers copy the parity back to the host.

A writer's round-``s`` put is ordered after its stream's round-``s - 1`` waits,
so every reader had put ``s - 1`` -- and, in its own stream order before that,
finished reading round ``s - 2``, the previous user of the same parity: no put
overwrites bytes a peer has not read.
"""

import time
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._utils import maybe_pin_memory

# Per-rank bytes one all-gather round carries (a broadcast round carries
# ``size * SM_FREE_CHUNK_BYTES``); larger payloads take more rounds. A group of
# size W maps 2 * W of these plus its sequence words.
SM_FREE_CHUNK_BYTES = 1 << 18

# Sequence values a call can stage without a host synchronization in between.
_SEQ_RING = 256

# A round that needs no SM finishes in about a millisecond; the bound only has
# to separate that from "waits for an SM", plus first-call setup.
SELF_TEST_TIMEOUT_S = 5.0
# After the saturating kernel is released every round must finish; past this
# bound the transport is unusable and initialization fails loudly.
SELF_TEST_DRAIN_TIMEOUT_S = 60.0


class _CudaFabric:
    """The four transport operations on a torch symmetric-memory buffer.

    ``put`` / ``put_word`` copy pinned host bytes to an offset of every rank's
    buffer (through the multicast mapping when the group has one),
    ``wait_words`` makes the current stream wait until every peer's sequence
    word reaches a value, and ``read`` copies a range of this rank's buffer to
    pinned host memory. All are enqueued on the current stream; ``sync`` waits
    for it and ``fence`` records an event on it.

    Built in two steps: ``__init__`` allocates on this rank only, ``connect``
    maps the peers' buffers (collective over the group), so a caller can agree
    on the local step before any rank enters the collective one.
    """

    def __init__(self, pg, device: torch.device, nbytes: int):
        import torch.distributed._symmetric_memory as symm_mem
        from cuda.bindings import driver
        from cuda.bindings import runtime as cudart

        self._symm_mem = symm_mem
        self._driver = driver
        self._cudart = cudart
        self._h2d = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        self._d2h = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        self._geq = driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
        self.device = device
        self.size = pg.size()
        self.rank = pg.rank()
        self._group_name = pg.group_name
        # Persistent, written in place by later calls: never inference tensors.
        with torch.inference_mode(False):
            self._buf = symm_mem.empty(nbytes, dtype=torch.uint8, device=device)
            self._buf.zero_()  # sequence words start at 0
        torch.cuda.synchronize(device)
        self._handle = None
        self._ptrs = []
        self._multicast = 0
        self._peers = [p for p in range(self.size) if p != self.rank]

    def connect(self) -> None:
        """Map every rank's buffer (collective over the group; host-blocking)."""
        self._symm_mem.enable_symm_mem_for_group(self._group_name)
        self._handle = self._symm_mem.rendezvous(self._buf, self._group_name)
        # buffer_ptrs are the bases of the symmetric blocks; the multicast
        # pointer already includes the tensor's offset into its block.
        base = [int(p) for p in self._handle.buffer_ptrs]
        offset = self._buf.data_ptr() - base[self.rank]
        self._ptrs = [p + offset for p in base]
        self._multicast = int(self._handle.multicast_ptr or 0)

    @staticmethod
    def local_problem(device: torch.device) -> Optional[str]:
        """Why this rank cannot run the fabric (checked before any collective step)."""
        try:
            import torch.distributed._symmetric_memory  # noqa: F401
            from cuda.bindings import driver
        except ImportError as e:
            return f"cannot import {e.name}"
        error, cu_device = driver.cuDeviceGet(device.index)
        supported = 0
        if error == driver.CUresult.CUDA_SUCCESS:
            error, supported = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                cu_device,
            )
        if error != driver.CUresult.CUDA_SUCCESS or not supported:
            return "no 64-bit CUDA stream memory operations"
        return None

    @property
    def multicast(self) -> bool:
        return bool(self._multicast)

    def _stream(self) -> int:
        return int(torch.cuda.current_stream(self.device).cuda_stream)

    def _copy(self, dst: int, src: int, nbytes: int, kind) -> None:
        status = self._cudart.cudaMemcpyAsync(dst, src, nbytes, kind, self._stream())[0]
        if status != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMemcpyAsync({nbytes} bytes) failed: {status!r}")

    def put(self, offset: int, src: int, nbytes: int) -> None:
        if self._multicast:
            self._copy(self._multicast + offset, src, nbytes, self._h2d)
            return
        for peer in self._peers:
            self._copy(self._ptrs[peer] + offset, src, nbytes, self._h2d)

    put_word = put

    def wait_words(self, offset: int, value: int) -> None:
        """Wait until every peer's word at ``offset + 8 * peer`` reaches ``value``."""
        stream = self._stream()
        base = self._ptrs[self.rank] + offset
        for peer in self._peers:
            (error,) = self._driver.cuStreamWaitValue64(stream, base + 8 * peer, value, self._geq)
            if error != self._driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuStreamWaitValue64 failed: {error!r}")

    def read(self, dst: int, offset: int, nbytes: int) -> None:
        self._copy(dst, self._ptrs[self.rank] + offset, nbytes, self._d2h)

    def sync(self) -> None:
        torch.cuda.current_stream(self.device).synchronize()

    def fence(self):
        event = torch.cuda.Event()
        event.record()
        return event


class SmFreeVoteStaging:
    """The round protocol of one vote group (see the module docstring).

    Buffer layout: ``size`` 8-byte sequence words, then two parities of
    ``size * chunk`` bytes. An all-gather round of ``n`` bytes per rank puts
    rank ``w``'s bytes at ``w * n`` of the parity; a broadcast round puts the
    root's bytes at the start of it. Calls are ordered on the caller's (side)
    stream and serialized by TorchDist's object-collective lock.
    """

    def __init__(
        self, pg, device: torch.device, chunk_bytes: int = SM_FREE_CHUNK_BYTES, fabric=None
    ):
        self.size = pg.size()
        self.rank = pg.rank()
        self.chunk = chunk_bytes
        self._words = -(-8 * self.size // 128) * 128
        nbytes = self._words + 2 * self.size * chunk_bytes
        self._fabric = fabric if fabric is not None else _CudaFabric(pg, device, nbytes)
        with torch.inference_mode(False):
            self._src = maybe_pin_memory(torch.zeros(self.size * chunk_bytes, dtype=torch.uint8))
            self._seq_ring = maybe_pin_memory(torch.zeros(_SEQ_RING, dtype=torch.int64))
            self._out = maybe_pin_memory(torch.zeros(self.size * chunk_bytes, dtype=torch.uint8))
        self._root_staging = None
        self._seq = 0
        self._pending = None  # fence over copies out of the host buffers

    @staticmethod
    def local_problem(device: torch.device) -> Optional[str]:
        return _CudaFabric.local_problem(device)

    def connect(self) -> None:
        """Map the peers' buffers: collective over the group (see ``_CudaFabric``)."""
        self._fabric.connect()

    @property
    def multicast(self) -> bool:
        return self._fabric.multicast

    def _host_buffers_free(self) -> None:
        # Copies out of the pinned buffers are asynchronous; a call that did not
        # synchronize leaves a fence the next host write must pass.
        if self._pending is not None:
            self._pending.synchronize()
            self._pending = None

    def _round(self, nbytes: int, put_offset: Optional[int], src: int) -> int:
        """Enqueue one round; returns the offset of its parity."""
        self._seq += 1
        seq = self._seq
        if seq % _SEQ_RING == 0:
            # The ring is about to wrap inside one call: drain it first.
            self._fabric.sync()
        parity = self._words + (seq & 1) * self.size * self.chunk
        if put_offset is not None and nbytes:
            self._fabric.put(parity + put_offset, src, nbytes)
        slot = seq % _SEQ_RING
        self._seq_ring[slot] = seq
        self._fabric.put_word(8 * self.rank, self._seq_ring.data_ptr() + 8 * slot, 8)
        self._fabric.wait_words(0, seq)
        return parity

    def allgather_words(self, value, dtype: torch.dtype) -> torch.Tensor:
        """All-gather one 8-byte scalar per rank; returns the CPU ``(W,)`` tensor.

        ``dtype`` is int64 or float64: the wire is the same 8 bytes.
        """
        self._host_buffers_free()
        self._src[:8].view(dtype)[0] = value
        parity = self._round(8, 8 * self.rank, self._src.data_ptr())
        self._fabric.read(self._out.data_ptr(), parity, 8 * self.size)
        self._fabric.sync()
        words = self._out[: 8 * self.size].view(dtype).clone()
        words[self.rank] = self._src[:8].view(dtype)[0]  # unicast puts skip this rank
        return words

    def allgather_bytes(
        self,
        local: Optional[torch.Tensor],
        local_size: int,
        max_size: int,
        read_back: bool = True,
    ) -> Optional[torch.Tensor]:
        """All-gather byte payloads of at most ``max_size`` bytes per rank.

        ``local`` is this rank's CPU uint8 payload (``local_size`` bytes, None
        when 0). Returns the CPU ``(W * max_size,)`` uint8 tensor, rank ``r``'s
        bytes at ``[r * max_size, r * max_size + size_r)``; ``read_back=False``
        runs the same rounds and returns None. Every rank passes the same
        ``max_size`` (it comes from a gathered size exchange), so every rank
        runs the same number of rounds.
        """
        world, chunk = self.size, self.chunk
        out = torch.empty((world, max_size), dtype=torch.uint8) if read_back else None
        for offset in range(0, max_size, chunk):
            n = min(chunk, max_size - offset)
            mine = max(0, min(n, local_size - offset))
            self._host_buffers_free()
            if mine:
                self._src[:mine].copy_(local[offset : offset + mine])
            parity = self._round(n, self.rank * n, self._src.data_ptr())
            if out is None:
                # The next round rewrites the host buffer behind this one's put.
                self._pending = self._fabric.fence()
                continue
            self._fabric.read(self._out.data_ptr(), parity, world * n)
            self._fabric.sync()
            out[:, offset : offset + n] = self._out[: world * n].view(world, n)
            if mine:
                out[self.rank, offset : offset + mine] = local[offset : offset + mine]
        return None if out is None else out.view(-1)

    def broadcast_bytes(self, data: Optional[bytes], root: int) -> Optional[bytes]:
        """Broadcast ``data`` from group rank ``root``; returns it on the others.

        The root passes its bytes and gets None back. It stages them in pinned
        memory and enqueues every round without waiting for them (the rounds'
        waits run on its stream), so it never holds its host on the call it is
        issuing; the next call's first host write passes a fence over those
        copies, so the root runs at most one call ahead of the slowest peer.
        """
        is_root = self.rank == root
        self._host_buffers_free()
        staging = None
        nbytes = len(data) if is_root else 0  # non-roots learn it in the first round
        if is_root:
            self._src[:8].view(torch.int64)[0] = nbytes
            staging = self._root_staging
            if nbytes and (staging is None or staging.numel() < nbytes):
                with torch.inference_mode(False):
                    staging = maybe_pin_memory(torch.empty(nbytes, dtype=torch.uint8))
                self._root_staging = staging
            if nbytes:
                staging[:nbytes].numpy()[:] = memoryview(data)
        size_parity = self._round(8, 8 * root if is_root else None, self._src.data_ptr())
        if not is_root:
            self._fabric.read(self._out.data_ptr(), size_parity + 8 * root, 8)
            self._fabric.sync()
            nbytes = int(self._out[:8].view(torch.int64)[0])
        per_round = self.size * self.chunk
        received = None if is_root else bytearray(nbytes)
        for offset in range(0, nbytes, per_round):
            n = min(per_round, nbytes - offset)
            src = staging.data_ptr() + offset if is_root else 0
            parity = self._round(n, 0 if is_root else None, src)
            if not is_root:
                self._fabric.read(self._out.data_ptr(), parity, n)
                self._fabric.sync()
                received[offset : offset + n] = self._out[:n].numpy().tobytes()
        if is_root:
            self._pending = self._fabric.fence()
            return None
        return bytes(received)

    def self_test_rounds(self) -> List[Callable[[], None]]:
        """Every kind of round this transport issues, without host synchronization."""
        world, chunk = self.size, self.chunk
        src = self._src.data_ptr()

        def words():
            parity = self._round(8, 8 * self.rank, src)
            self._fabric.read(self._out.data_ptr(), parity, 8 * world)

        def payload():
            parity = self._round(chunk, self.rank * chunk, src)
            self._fabric.read(self._out.data_ptr(), parity, world * chunk)

        def broadcast():
            # The root's bytes come from the host buffer it owns: world * chunk.
            parity = self._round(world * chunk, 0 if self.rank == 0 else None, src)
            if self.rank:
                self._fabric.read(self._out.data_ptr(), parity, world * chunk)

        return [words, payload, broadcast]


class SmSaturator:
    """Holds every SM's thread slots with spinning programs until released.

    ``max_threads_per_multi_processor / 512`` programs of 512 threads per SM:
    no further CTA of any kernel fits while they run, while copy engines and
    stream memory operations still progress.
    """

    THREADS = 512

    def __init__(self, device: torch.device):
        props = torch.cuda.get_device_properties(device)
        per_sm = max(1, props.max_threads_per_multi_processor // self.THREADS)
        self.grid = props.multi_processor_count * per_sm
        self.device = device
        self.kernel = _spin_kernel()
        with torch.inference_mode(False):
            self.flag = torch.ones(1, dtype=torch.int32, device=device)
            self.resident = torch.zeros(self.grid, dtype=torch.int32, device=device)
            # Copy-engine transfers either way; pinned only makes them async.
            self.one_host = maybe_pin_memory(torch.ones(1, dtype=torch.int32))
            self.resident_host = maybe_pin_memory(torch.zeros(self.grid, dtype=torch.int32))
        self.spin_stream = torch.cuda.Stream(device)
        # The release must never queue behind work that needs an SM: a
        # high-priority stream comes from a separate pool, so it cannot be the
        # CUDA stream the default-priority vote stream maps to.
        self.copy_stream = torch.cuda.Stream(device, priority=-1)
        # Compile and load now (flag = 1: the programs exit at once).
        self._launch()
        torch.cuda.synchronize(device)

    def _launch(self):
        with torch.cuda.stream(self.spin_stream):
            self.kernel[(self.grid,)](self.flag, self.resident, num_warps=self.THREADS // 32)

    def saturate(self, timeout_s: float = 5.0) -> int:
        """Launch the spinning grid; returns how many programs became resident
        (``grid`` once every SM is held, fewer if others hold SMs)."""
        self.flag.zero_()
        self.resident.zero_()
        torch.cuda.synchronize(self.device)
        self._launch()
        deadline = time.monotonic() + timeout_s
        resident = 0
        while time.monotonic() < deadline:
            with torch.cuda.stream(self.copy_stream):
                # A copy-engine read: progresses while the SMs are held.
                self.resident_host.copy_(self.resident, non_blocking=True)
            self.copy_stream.synchronize()
            resident = int(self.resident_host.sum())
            if resident == self.grid:
                break
            time.sleep(0.002)
        return resident

    def release(self):
        with torch.cuda.stream(self.copy_stream):
            self.flag.copy_(self.one_host, non_blocking=True)


def _spin_kernel():
    """Triton kernel holding its SM until ``flag`` turns nonzero (built lazily:
    Triton is only needed by the self-test)."""
    import triton
    import triton.language as tl

    @triton.jit
    def spin(flag_ptr, resident_ptr):
        # One slot per program: an exact residency count.
        tl.store(resident_ptr + tl.program_id(0), 1)
        value = tl.load(flag_ptr, volatile=True)
        while value == 0:
            value = tl.load(flag_ptr, volatile=True)

    return spin


def drain(
    stream: torch.cuda.Stream, what: str, timeout_s: float = SELF_TEST_DRAIN_TIMEOUT_S
) -> None:
    """Wait for ``stream`` with a bound: a vote round that never completes would
    otherwise leave the caller (and every later device-wide sync) blocked."""
    with torch.cuda.stream(stream):
        done = torch.cuda.Event()
        done.record()
    deadline = time.monotonic() + timeout_s
    while not done.query():
        if time.monotonic() > deadline:
            raise RuntimeError(f"TorchDist: {what} did not finish within {timeout_s:.0f} s")
        time.sleep(0.001)


def verify_sm_free(
    rounds: Sequence[Callable[[], None]],
    stream: torch.cuda.Stream,
    saturator: SmSaturator,
    world_barrier: Callable[[], None],
) -> Tuple[bool, str]:
    """Run ``rounds`` on ``stream`` while ``saturator`` holds every SM.

    Returns whether they finished while the SMs were held, and what this rank
    saw. Collective: every rank of the world calls it with the same rounds.
    ``world_barrier`` must not use the GPU (it runs while the SMs are held).
    Raises if the rounds do not finish even after the SMs are released: the
    vote transport would be unusable.
    """
    device = saturator.device
    # torch hands streams out of a 32-entry round-robin pool: a spinning grid
    # on the vote stream itself would hold the rounds back.
    while saturator.spin_stream.cuda_stream == stream.cuda_stream:
        saturator.spin_stream = torch.cuda.Stream(device)
    try:
        resident = saturator.saturate()
        world_barrier()  # every rank holds its SMs before any round is issued
        t0 = time.monotonic()
        with torch.cuda.stream(stream):
            for issue in rounds:
                issue()
            done = torch.cuda.Event()
            done.record()
        finished = False
        deadline = t0 + SELF_TEST_TIMEOUT_S
        while time.monotonic() < deadline:
            if done.query():
                finished = True
                break
            time.sleep(0.0005)
        elapsed_ms = (time.monotonic() - t0) * 1e3
    finally:
        # On every exit: a grid left spinning wedges the GPU for good.
        saturator.release()
    deadline = time.monotonic() + SELF_TEST_DRAIN_TIMEOUT_S
    while not done.query():
        if time.monotonic() > deadline:
            raise RuntimeError(
                "TorchDist: the SM-free vote self-test did not finish within "
                f"{SELF_TEST_DRAIN_TIMEOUT_S:.0f} s after the SMs were released"
            )
        time.sleep(0.001)
    torch.cuda.synchronize(device)
    held = f"{resident}/{saturator.grid} spinning programs resident"
    if resident != saturator.grid:
        return False, f"could not hold every SM ({held})"
    if not finished:
        return False, f"not finished within {SELF_TEST_TIMEOUT_S:.0f} s while {held}"
    return True, f"finished in {elapsed_ms:.1f} ms while {held}"


def host_reduce(values: List[float], op_name: str) -> float:
    """Reduce gathered scalars on the host (the SM-free scalar all-reduce)."""
    if op_name == "SUM":
        return float(sum(values))
    if op_name == "PRODUCT":
        result = 1.0
        for value in values:
            result *= value
        return result
    if op_name == "MIN":
        return float(min(values))
    if op_name == "MAX":
        return float(max(values))
    raise ValueError(f"no host reduction for {op_name}")
