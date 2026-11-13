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
"""Direct peer copy of the NIXL transfer agent (TRTLLM_NIXL_DIRECT_COPY=1), on one node.

A receiver process allocates its registered memory the way KVCacheManagerV2 does (cuMemCreate chunks with
CU_MEM_HANDLE_TYPE_FABRIC mapped back to back in one reservation, two regions), a sender process on another GPU
(each process sees only its own GPU, as in a disaggregated deployment) writes a descriptor list: either mixed
(aligned pages, sub-16-byte and odd sizes, descriptors larger than one kernel work unit) or head-slice sized, some
crossing physical chunk boundaries. The sender imports the receiver's memory in the background; it waits for the
mapping before the writes it checks, and one case writes before the import is done and reloads the peer meanwhile.
The receiver checks every byte (and that bytes outside the descriptors stay zero); the sender checks which path
served the writes. The cross-node path is the same code with a peer on another host of the NVLink domain.
"""

import multiprocessing as mp
import os
import random

import numpy as np
import pytest

# Force a deterministic UCX/NIXL config regardless of what the cluster/CI injects (a job-level UCX_TLS=tcp makes the
# NIXL UCX backend refuse to start on GPU nodes); see test_kv_transfer.py for the full rationale.
os.environ["UCX_TLS"] = "^ib,gdr_copy"
os.environ["TRTLLM_NIXL_NUM_THREADS"] = "1"

_CHUNK = 2 << 20
_REGION_CHUNKS = (6, 3)


def _driver():
    try:
        from cuda.bindings import driver as cuda
    except ImportError:  # older cuda-python
        from cuda import cuda
    return cuda


def _check(result):
    err = result[0] if isinstance(result, tuple) else result
    cuda = _driver()
    if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    if isinstance(result, tuple):
        return result[1] if len(result) == 2 else result[1:]
    return None


def _alloc_fabric_region(n_chunks: int, device: int):
    cuda = _driver()
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    size = n_chunks * _CHUNK
    va = _check(cuda.cuMemAddressReserve(size, _CHUNK, 0, 0))
    access = cuda.CUmemAccessDesc()
    access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device
    access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    for i in range(n_chunks):
        handle = _check(cuda.cuMemCreate(_CHUNK, prop, 0))
        _check(cuda.cuMemMap(int(va) + i * _CHUNK, _CHUNK, 0, handle, 0))
        _check(cuda.cuMemRelease(handle))
        _check(cuda.cuMemSetAccess(int(va) + i * _CHUNK, _CHUNK, [access], 1))
    _check(cuda.cuMemsetD8(int(va), 0, size))
    return int(va), size


def _plan(seed: int, regions: list[tuple[int, int]], src_size: int, kind: str = "mixed"):
    """Descriptors (src_off, dst_addr, len), destinations non-overlapping, some crossing chunk boundaries.

    ``mixed`` spans sub-16-byte to multi-unit sizes (one CTA per kernel work unit); ``slices`` is mostly 8 KiB
    descriptors like per-layer KV head slices, plus a few odd sizes (one warp per work unit).
    """
    rng = random.Random(seed)
    if kind == "slices":
        sizes = [8192] * 12 + [4608, 1000, 7, 384]
    else:
        sizes = [4096, 65536, 163840, 300000 & ~15, 1 << 20, 3 << 20, 1, 7, 1000, 4100, 16, 48]
    descs = []
    for base, size in regions:
        cursor = 0
        while True:
            length = rng.choice(sizes)
            gap = rng.choice([0, 0, 16, 4096, 12345])
            start = cursor + gap
            # every so often, place the descriptor across the next chunk boundary
            if rng.random() < 0.25:
                boundary = (start // _CHUNK + 1) * _CHUNK
                start = max(start, boundary - length // 2)
            if start + length > size:
                break
            src_off = rng.randrange(0, src_size - length)
            if length % 16 == 0 and rng.random() < 0.7:
                src_off &= ~15
            descs.append((src_off, base + start, length))
            cursor = start + length
    rng.shuffle(descs)
    return descs


def _receiver(conn, device: str, direct: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["TRTLLM_NIXL_DIRECT_COPY"] = "0" if direct == "nixl" else "1"
    import torch

    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent

    try:
        torch.cuda.set_device(0)
        torch.zeros(1, device="cuda")
        try:
            regions = [_alloc_fabric_region(n, 0) for n in _REGION_CHUNKS]
        except RuntimeError as e:
            conn.send(("skip", f"fabric memory unavailable: {e}"))
            return
        agent = BindingsNixlTransferAgent("rx", num_threads=1)
        agent.register_memory(
            RegMemoryDescs("VRAM", [(va, size, 0, f"r{i}") for i, (va, size) in enumerate(regions)])
        )
        conn.send(("ready", agent.get_local_agent_desc(), regions))
        msg = conn.recv()
        assert msg[0] == "load"
        agent.load_remote_agent("tx", msg[1])
        conn.send(("loaded",))
        msg = conn.recv()
        assert msg[0] == "verify"
        _, descs, src = msg
        cuda = _driver()
        bad = 0
        for va, size in regions:
            host = np.empty(size, dtype=np.uint8)
            _check(cuda.cuMemcpyDtoH(host.ctypes.data, va, size))
            expect = np.zeros(size, dtype=np.uint8)
            for src_off, dst, length in descs:
                if va <= dst < va + size:
                    expect[dst - va : dst - va + length] = src[src_off : src_off + length]
            bad += int(np.count_nonzero(host != expect))
        conn.send(("result", bad))
        agent.invalidate_remote_agent("tx")
        agent.shutdown()
    except Exception as e:  # report instead of hanging the parent
        conn.send(("error", repr(e)))


def _sender(conn, device: str, direct: str, rx_desc, regions, seed: int, plan: str, early: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["TRTLLM_NIXL_DIRECT_COPY"] = "0" if direct == "nixl" else "1"
    import time

    import torch

    from tensorrt_llm._torch.disaggregation.base.agent import (
        MemoryDescs,
        MemoryType,
        RegMemoryDescs,
        TransferOp,
        TransferRequest,
    )
    from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import BindingsNixlTransferAgent

    try:
        torch.cuda.set_device(0)
        src_size = 8 << 20
        gen = torch.Generator(device="cpu").manual_seed(seed)
        src_host = torch.randint(0, 256, (src_size,), dtype=torch.uint8, generator=gen)
        src = src_host.cuda()
        agent = BindingsNixlTransferAgent("tx", num_threads=1)
        agent.register_memory(RegMemoryDescs("VRAM", [(src.data_ptr(), src_size, 0, "src")]))
        agent.load_remote_agent("rx", rx_desc)
        conn.send(("desc", agent.get_local_agent_desc()))
        assert conn.recv()[0] == "go"
        descs = _plan(seed, regions, src_size, plan)
        base = src.data_ptr()

        def write() -> bool:
            return agent.submit_transfer_requests(
                TransferRequest(
                    TransferOp.WRITE,
                    MemoryDescs(MemoryType.VRAM, [(base + s, n, 0) for s, _, n in descs]),
                    MemoryDescs(MemoryType.VRAM, [(d, n, 0) for _, d, n in descs]),
                    "rx",
                    None,
                )
            ).wait()

        oks = []
        if early:
            # Right after the load the import may still be running: this write takes either path. Then drop the
            # peer while its import may be in flight and load it again; only the second import may be published.
            oks.append(write())
            agent.invalidate_remote_agent("rx")
            agent.load_remote_agent("rx", rx_desc)
        if direct != "nixl":
            deadline = time.monotonic() + 120
            while agent.direct_mapped_peer_count < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
        before = agent.direct_submit_count
        # Two writes of the same plan once the peer is mapped: the second reuses the copy context.
        oks += [write(), write()]
        mapped_writes = agent.direct_submit_count - before
        conn.send(("done", all(oks), mapped_writes, descs, src_host.numpy()))
        assert conn.recv()[0] == "bye"
        agent.invalidate_remote_agent("rx")
        agent.shutdown()
    except Exception as e:
        conn.send(("error", repr(e)))


def _expect(conn, kind):
    msg = conn.recv()
    if msg[0] == "error":
        raise AssertionError(f"child failed: {msg[1]}")
    if msg[0] == "skip":
        pytest.skip(msg[1])
    assert msg[0] == kind, msg
    return msg


@pytest.mark.parametrize(
    "direct,plan,early",
    [
        ("kernel", "mixed", False),
        # Head-slice-sized descriptors (warp-per-unit kernel), written while the import may still be running, and a
        # drop + reload of the peer around its background import.
        ("kernel", "slices", True),
        ("nixl", "mixed", False),
    ],
)
def test_direct_peer_copy_matches_descriptors(direct, plan, early):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorrt_llm.tensorrt_llm_transfer_agent_binding")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    n_gpus = torch.cuda.device_count()
    rx_dev, tx_dev = ("1", "0") if n_gpus >= 2 else ("0", "0")
    ctx = mp.get_context("spawn")
    rx_conn, rx_child = ctx.Pipe()
    tx_conn, tx_child = ctx.Pipe()
    rx = ctx.Process(target=_receiver, args=(rx_child, rx_dev, direct))
    rx.start()
    tx = None
    try:
        _, rx_desc, regions = _expect(rx_conn, "ready")
        tx = ctx.Process(
            target=_sender, args=(tx_child, tx_dev, direct, rx_desc, regions, 1234, plan, early)
        )
        tx.start()
        _, tx_desc = _expect(tx_conn, "desc")
        rx_conn.send(("load", tx_desc))
        _expect(rx_conn, "loaded")
        tx_conn.send(("go",))
        _, ok, mapped_writes, descs, src = _expect(tx_conn, "done")
        assert ok, "transfer failed"
        rx_conn.send(("verify", descs, src))
        _, bad = _expect(rx_conn, "result")
        tx_conn.send(("bye",))
        assert bad == 0, f"{bad} bytes differ from the descriptors' sources"
        expected = 0 if direct == "nixl" else 2
        assert mapped_writes == expected, (
            f"expected {expected} of the writes after the mapping on the direct path, got {mapped_writes}"
        )
        assert any(d[2] % 16 for d in descs)
        assert any(d[1] // _CHUNK != (d[1] + d[2] - 1) // _CHUNK for d in descs)
        if plan == "mixed":
            assert any(d[2] > (256 << 10) for d in descs)
        else:
            assert sum(d[2] for d in descs) <= len(descs) * (32 << 10)
    finally:
        for p in (tx, rx):
            if p is not None:
                p.join(timeout=120)
                if p.is_alive():
                    p.kill()
