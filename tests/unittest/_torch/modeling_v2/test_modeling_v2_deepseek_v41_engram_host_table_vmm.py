# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ``vmm`` backend of the DeepSeek-V4.1 target's ``engram_host_table.py``.

The backend keeps DeepSeek-V4.1's Engram tables in CUDA VMM host-NUMA allocations
(``cuMemCreate`` on the CPU sockets, 1 GiB slabs alternating between them), shared
between processes as POSIX file descriptors over a node-local abstract Unix socket,
and hands ``engram_gather`` CUDA views of that host memory.

What is checked, on a synthetic two-layer checkpoint (never the real 202.76 GB table):

* the creator's views are CUDA views of host memory: their bytes equal the
  checkpoint's, the manifest's chunk CRCs re-verify from the host view, and no
  device memory holds a copy;
* ``engram_gather`` through them equals the torch reference bit for bit, eager and
  replayed from a CUDA graph, for both layers;
* a second process imports the table from the creator's socket and gathers the
  same bits; after the creator closes, a third process imports it from the second
  (the table stays reachable while any holder lives);
* a creator that sees one GPU (``CUDA_VISIBLE_DEVICES`` of one device, for a device on
  each CPU socket) places its slabs only on the sockets it may allocate on, and a
  process that sees only a GPU of the other socket imports the table and gathers the
  same bits;
* host out-of-memory at create (injected ``CUDA_ERROR_OUT_OF_MEMORY``): the create
  releases what it allocated and retries with backoff, and succeeds when memory
  returns; when it never does, the table falls back to the shm backend with a
  warning (every local rank then attaches the shm file), unless the backend is
  set to ``vmm`` explicitly, where the create fails after joining the barrier;
* the backend switch and the per-checkpoint socket name.
"""

import json
import os
import struct
import uuid

import pytest
import torch
import torch.multiprocessing as mp

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    engram_host_table as H,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.embedding.engram_gather import (  # noqa: E501
    EngramGather,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# The target's gather entry, unbound: every call passes its operands.
engram_gather = EngramGather()

ROWS = 100_003
DIM = 256


def _table(rows: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (rows, DIM), dtype=torch.uint8, generator=g)
    w[(w == 0x7F) | (w == 0xFF)] = 0  # E4M3 NaN codes: NaN payloads differ between Triton and torch
    s = torch.randint(118, 124, (rows, DIM // 32), dtype=torch.uint8, generator=g)
    return w.view(torch.float8_e4m3fn), s


def _ref(ids, w, s):
    ids = ids.cpu()
    vals = w[ids].float().unflatten(-1, (DIM // 32, 32))
    scale = s[ids].view(torch.float8_e8m0fnu).float().unsqueeze(-1)
    return (vals * scale).flatten(-2).to(torch.bfloat16)


def _write_checkpoint(ckpt_dir: str, tables) -> None:
    header, blobs, at = {}, [], 0
    for layer, (w, s) in tables.items():
        for kind, t, dtype in (("weight", w, "F8_E4M3"), ("scale", s, "U8")):
            b = t.view(torch.uint8).contiguous().numpy().tobytes()
            header[f"layers.{layer}.engram.embed.{kind}"] = {
                "dtype": dtype,
                "shape": list(t.shape),
                "data_offsets": [at, at + len(b)],
            }
            blobs.append(b)
            at += len(b)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    shard = "model-00001-of-00001.safetensors"
    with open(os.path.join(ckpt_dir, shard), "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": {k: shard for k in header}}, f)


def _ids(tokens: int, seed: int, device) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, ROWS, (tokens, 24), generator=g)
    ids[0, :4] = torch.tensor([0, 1, ROWS - 2, ROWS - 1])
    return ids.to(device)


def _gather_all(table, device, seeds=(1, 2)):
    out = {}
    for layer in table.layers:
        w, s = table.weight(layer), table.scale(layer).view(torch.uint8)
        for seed in seeds:
            ids = _ids(64, seed * 100 + layer, device)
            out[(layer, seed)] = engram_gather(ids, w, s, 0, w.shape[0]).cpu()
    return out


def _child(rank, ckpt, name, device_index, queue, release):
    """Import the table (local_rank 1: no fill), gather, report, keep holding (and serving) until released."""
    torch.cuda.set_device(device_index)
    dev = torch.device("cuda", device_index)
    torch.zeros(1, device=dev)
    table = H.open_vmm_engram_table(ckpt, local_rank=1, device=dev, wait_s=30.0, name=name)
    table.check_gather_views(dev)
    got = _gather_all(table, dev)
    queue.put((rank, {"created": table.created, "gathers": got}))
    release.wait(timeout=1800)
    table.close()


def _creator(ckpt, name, queue, release):
    """Create the table as local rank 0 on this process's only visible device, report, hold it until released."""
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    torch.zeros(1, device=dev)
    table = H.open_vmm_engram_table(
        ckpt, local_rank=0, device=dev, headroom_bytes=1 << 30, name=name, threads=4
    )
    table.check_gather_views(dev)
    fill = table.manifest["fill"]
    queue.put(
        (
            "creator",
            {
                "created": table.created,
                "numa_bytes": table.numa_bytes(),
                "refused": {int(k): v for k, v in fill["numa_nodes_refused"].items()},
                "device_host_nodes": {int(k): v for k, v in fill["device_host_nodes"].items()},
                "gathers": _gather_all(table, dev),
            },
        )
    )
    release.wait(timeout=1800)
    table.close()


def _physical(logical: int) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    return visible.split(",")[logical].strip() if visible else str(logical)


def _start(ctx, target, args, visible: str):
    """Start a spawned process that sees only ``visible`` (it inherits the environment at start)."""
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    try:
        proc = ctx.Process(target=target, args=args)
        proc.start()
    finally:
        if old is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old
    return proc


@pytest.fixture
def checkpoint(tmp_path):
    tables = {1: _table(ROWS, 7001), 14: _table(ROWS, 7014)}
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_checkpoint(str(ckpt), tables)
    return str(ckpt), tables


def test_backend_switch(monkeypatch):
    monkeypatch.delenv(H.BACKEND_ENV, raising=False)
    assert H.host_table_backend() == "vmm"
    for value, want in (("shm", "shm"), (" VMM ", "vmm")):
        monkeypatch.setenv(H.BACKEND_ENV, value)
        assert H.host_table_backend() == want
    monkeypatch.setenv(H.BACKEND_ENV, "hugetlb")
    with pytest.raises(ValueError):
        H.host_table_backend()


def test_socket_name_depends_on_the_checkpoint_bytes(checkpoint, tmp_path):
    ckpt, tables = checkpoint
    other = tmp_path / "other"
    other.mkdir()
    _write_checkpoint(str(other), {1: _table(ROWS + 1, 1), 14: _table(ROWS, 2)})
    a = H.default_vmm_socket_name(H._layout(H.find_engram_table_sources(ckpt)))
    b = H.default_vmm_socket_name(H._layout(H.find_engram_table_sources(str(other))))
    assert a != b and a.startswith(f"trtllm-dsv41-engram-{os.getuid()}.vmm.")


def test_create_gather_graph_import_and_takeover(checkpoint):
    ckpt, tables = checkpoint
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    name = f"pf-vmm-test-{uuid.uuid4().hex[:12]}"
    torch.cuda.synchronize()
    free0, _ = torch.cuda.mem_get_info(dev)
    alloc0 = torch.cuda.memory_allocated(dev)
    table = H.open_vmm_engram_table(
        ckpt, local_rank=0, device=dev, headroom_bytes=1 << 30, name=name, threads=4
    )
    importers = []
    try:
        assert table.created and not table.fill_report.reused
        table.check_gather_views(dev)
        # CUDA views of host memory, bytes equal to the checkpoint's, CRCs re-verify, no device copy.
        for layer, (w, s) in tables.items():
            vw, vs = table.weight(layer), table.scale(layer)
            assert (
                vw.is_cuda
                and vw.device == dev
                and tuple(vw.shape) == (ROWS, DIM)
                and vw.dtype == torch.float8_e4m3fn
            )
            # the scale's dtype is the checkpoint's (U8 here, F8_E8M0 in the real checkpoint)
            assert vs.is_cuda and vs.dtype == torch.uint8 and tuple(vs.shape) == (ROWS, DIM // 32)
            hw = torch.frombuffer(
                table.host_buffer(f"layers.{layer}.engram.embed.weight"), dtype=torch.uint8
            )
            hs = torch.frombuffer(
                table.host_buffer(f"layers.{layer}.engram.embed.scale"), dtype=torch.uint8
            )
            assert torch.equal(hw.view(ROWS, DIM), w.view(torch.uint8))
            assert torch.equal(hs.view(ROWS, DIM // 32), s)
        checked, bad = table.verify(full=True)
        assert checked > 0 and bad == []
        assert sum(table.numa_bytes().values()) == table.allocated_bytes >= table.nbytes
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated(dev) == alloc0
        assert free0 - torch.cuda.mem_get_info(dev)[0] < 64 << 20, (
            "the table must not occupy device memory"
        )

        # Gathers: eager and replayed from a CUDA graph, bit-exact against the reference.
        static = _ids(48, 11, dev)
        pair = {
            layer: (table.weight(layer), table.scale(layer).view(torch.uint8)) for layer in tables
        }
        for layer, (w, s) in pair.items():
            engram_gather(static, w, s, 0, ROWS)  # warm-up outside capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = {layer: engram_gather(static, w, s, 0, ROWS) for layer, (w, s) in pair.items()}
        for seed in (12, 13):
            static.copy_(_ids(48, seed, dev))
            graph.replay()
            torch.cuda.synchronize()
            for layer, (w, s) in tables.items():
                ref = _ref(static, w, s)
                assert torch.equal(outs[layer].cpu().view(torch.int16), ref.view(torch.int16))
                eager = engram_gather(static, *pair[layer], 0, ROWS)
                assert torch.equal(eager.view(torch.int16), outs[layer].view(torch.int16))
        del graph, outs
        mine = _gather_all(table, dev)
        for (layer, seed), got in mine.items():
            ref = _ref(_ids(64, seed * 100 + layer, "cpu"), *tables[layer])
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16))

        # A second process imports from this one; after this one closes, a third imports from the second.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        release = ctx.Event()
        other = 1 if torch.cuda.device_count() > 1 else 0
        importers.append(ctx.Process(target=_child, args=(1, ckpt, name, other, queue, release)))
        importers[-1].start()
        rank, first = queue.get(timeout=1200)
        assert rank == 1 and not first["created"]
        for key, got in first["gathers"].items():
            assert torch.equal(got.view(torch.int16), mine[key].view(torch.int16)), key
        table.close()  # the creator stops serving; the importer's standby server takes the name over
        importers.append(ctx.Process(target=_child, args=(2, ckpt, name, 0, queue, release)))
        importers[-1].start()
        rank, second = queue.get(timeout=1200)
        assert rank == 2 and not second["created"]
        for key, got in second["gathers"].items():
            assert torch.equal(got.view(torch.int16), mine[key].view(torch.int16)), key
        release.set()
        for p in importers:
            p.join(timeout=120)
            assert p.exitcode == 0
    finally:
        table.close()
        for p in importers:
            if p.is_alive():
                p.kill()


def test_single_gpu_creator_on_each_socket_and_cross_socket_import(checkpoint):
    ckpt, tables = checkpoint
    torch.zeros(1, device="cuda")
    host_nodes = H._visible_device_host_nodes()
    by_node = {}
    for index, node in sorted(host_nodes.items()):
        by_node.setdefault(node, index)
    creators = sorted(by_node.items())  # one device per CPU socket that has GPUs
    ctx = mp.get_context("spawn")
    for node, index in creators:
        others = [i for n, i in creators if n != node]
        name = f"pf-vmm-sock-{uuid.uuid4().hex[:12]}"
        queue, release = ctx.Queue(), ctx.Event()
        procs = [_start(ctx, _creator, (ckpt, name, queue, release), _physical(index))]
        try:
            role, made = queue.get(timeout=1200)
            assert role == "creator" and made["created"]
            assert made["device_host_nodes"] == {0: node}, made["device_host_nodes"]
            slab_nodes = set(made["numa_bytes"])
            assert slab_nodes and not slab_nodes & set(made["refused"]), made
            if node >= 0:
                assert node in slab_nodes, made
            for (layer, seed), got in made["gathers"].items():
                ref = _ref(_ids(64, seed * 100 + layer, "cpu"), *tables[layer])
                assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (
                    node,
                    layer,
                    seed,
                )
            print(
                f"\ncreator on GPU {_physical(index)} (host node {node}): slabs on {sorted(slab_nodes)}, "
                f"refused {made['refused']}"
            )
            for rank, other in enumerate(others, start=1):
                procs.append(
                    _start(ctx, _child, (rank, ckpt, name, 0, queue, release), _physical(other))
                )
                got_rank, imported = queue.get(timeout=1200)
                assert got_rank == rank and not imported["created"]
                for key, got in imported["gathers"].items():
                    assert torch.equal(
                        got.view(torch.int16), made["gathers"][key].view(torch.int16)
                    ), (node, other, key)
                print(
                    f"importer on GPU {_physical(other)} (host node {host_nodes[other]}): gathers equal"
                )
            release.set()
            for p in procs:
                p.join(timeout=120)
                assert p.exitcode == 0
        finally:
            release.set()
            for p in procs:
                if p.is_alive():
                    p.kill()


class _InjectOom:
    """Wraps ``_cu_mem_create``: the listed calls (1-based) return ``CUDA_ERROR_OUT_OF_MEMORY``
    (``always``: every call); records the handle objects created and released, compared by
    identity (the driver reuses a handle's value once it is released)."""

    def __init__(self, monkeypatch, fail_calls=(), always=False):
        from cuda.bindings import driver as cu

        self.cu = cu
        self.fail_calls, self.always = set(fail_calls), always
        self.calls, self.created, self.released = 0, [], []
        self._lock = __import__("threading").Lock()
        real_create, real_release = H._cu_mem_create, cu.cuMemRelease

        def create(size, node):
            with self._lock:
                self.calls += 1
                fail = self.always or self.calls in self.fail_calls
            if fail:
                return cu.CUresult.CUDA_ERROR_OUT_OF_MEMORY, None
            result = real_create(size, node)
            if result[0] == cu.CUresult.CUDA_SUCCESS:
                with self._lock:
                    self.created.append(result[1])
            return result

        def release(handle):
            with self._lock:
                self.released.append(handle)
            return real_release(handle)

        monkeypatch.setattr(H, "_cu_mem_create", create)
        monkeypatch.setattr(cu, "cuMemRelease", release)

    def was_released(self, handle) -> bool:
        return any(handle is r for r in self.released)


def test_create_retries_after_host_oom(checkpoint, monkeypatch):
    ckpt, tables = checkpoint
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    monkeypatch.setenv(H.CREATE_RETRY_ENV, "60")
    # One allocation thread, so the 4 slabs are created in order: attempt 1 creates slab 0 and fails
    # on slab 1 (calls 1-2), attempt 2 the same (calls 3-4), attempt 3 creates all four (calls 5-8).
    oom = _InjectOom(monkeypatch, fail_calls=(2, 4))
    name = f"pf-vmm-oom-{uuid.uuid4().hex[:12]}"
    table = H.open_vmm_engram_table(
        ckpt, local_rank=0, device=dev, headroom_bytes=1 << 30, name=name, threads=1
    )
    try:
        attempts = table.manifest["fill"]["create_attempts"]
        assert attempts == 3, attempts
        assert oom.calls == 8 and len(oom.created) == 6, (oom.calls, len(oom.created))
        failed, kept = oom.created[:2], oom.created[2:]
        assert all(oom.was_released(h) for h in failed), "a failed attempt's handles are released"
        # (the NUMA probe's one-granule allocations are released too; count only the slabs)
        assert sum(any(r is c for c in oom.created) for r in oom.released) == 2
        assert not any(oom.was_released(h) for h in kept)
        assert len(table._handles) == 4 and all(any(h is k for k in kept) for h in table._handles)
        for (layer, seed), got in _gather_all(table, dev).items():
            ref = _ref(_ids(64, seed * 100 + layer, "cpu"), *tables[layer])
            assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (layer, seed)
        print(
            f"\ncreate succeeded on attempt {attempts}; the 2 slabs of the failed attempts were released"
        )
    finally:
        table.close()
    assert all(oom.was_released(h) for h in oom.created)


def test_create_oom_falls_back_to_shm_unless_vmm_is_explicit(checkpoint, monkeypatch):
    ckpt, tables = checkpoint
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    shm_path = f"/dev/shm/pf-vmm-oom-fallback-{uuid.uuid4().hex[:12]}.bin"
    monkeypatch.setenv(H.PATH_ENV, shm_path)  # also makes the lock and the socket name unique
    monkeypatch.setenv(H.CREATE_RETRY_ENV, "2")
    monkeypatch.delenv(H.BACKEND_ENV, raising=False)
    oom = _InjectOom(monkeypatch, always=True)
    barriers = []
    try:
        with pytest.warns(RuntimeWarning, match="falling back to the shm backend"):
            rank0 = H.open_shared_engram_table(
                ckpt,
                local_rank=0,
                barrier=lambda: barriers.append(0),
                device=dev,
                threads=4,
                headroom_bytes=1 << 30,
            )
        # another local rank, after the barrier: nobody serves the VMM table, the shm file is complete
        rank1 = H.open_shared_engram_table(
            ckpt, local_rank=1, barrier=lambda: barriers.append(1), device=dev
        )
        assert barriers == [0, 1] and oom.calls >= 2 and oom.created == []
        for table in (rank0, rank1):
            assert isinstance(table, H.SharedEngramHostTable) and table.path == shm_path, type(
                table
            )
            table.check_gather_views(dev)
            for layer, (w, s) in tables.items():
                assert torch.equal(table.weight(layer).view(torch.uint8), w.view(torch.uint8))
                assert torch.equal(table.scale(layer).view(torch.uint8), s)
        assert rank0.fill_report is not None and not rank0.fill_report.reused
        rank1.close()
        rank0.close()

        # explicit vmm: no fallback; rank 0 joins the barrier, then raises
        monkeypatch.setenv(H.BACKEND_ENV, "vmm")
        barriers.clear()
        with pytest.raises(H.VmmCreateError, match="could not be allocated"):
            H.open_shared_engram_table(
                ckpt,
                local_rank=0,
                barrier=lambda: barriers.append(0),
                device=dev,
                threads=4,
                headroom_bytes=1 << 30,
            )
        assert barriers == [0]
    finally:
        H.remove_shared_engram_table(shm_path, force=True)
        for suffix in (".lock", ".owner", ".vmm.lock"):
            if os.path.exists(shm_path + suffix):
                os.unlink(shm_path + suffix)
