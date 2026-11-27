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
"""Two-GPU, two-process test of the KV fabric batch writer against a real NIXL transfer agent.

The receiver process owns FABRIC VMM pools (32 MiB chunks, VA contiguous, like the KV v2 pool),
exports them with ``fabric_writer.export_pools`` and registers them with its NIXL agent. The sender
process drives a real ``Sender``. The receiver verifies every written fragment byte by byte and
checks every other byte is zero.
"""

import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

try:
    import pytest
except ImportError:  # direct runs do not need pytest
    pytest = None

CH = 32 << 20
SENDER_DEV, OWNER_DEV = 0, 1
PEER = "ut_dec"
EXPECTED_TESTS = [
    "fbw_concurrent_byte_exact",
    "fbw_equals_nixl",
    "gate_routing",
    "export_with_hole",
    "fault_first_batch",
    "fault_later_batch",
]

_M1 = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
_M2 = -4658895280553007687  # 0xBF58476D1CE4E5B9 as int64


def gen(torch, seed: int, nbytes: int, dev: int):
    """Deterministic integer-hash payload (identical on any GPU), ``nbytes`` multiple of 8."""
    v = torch.arange(nbytes // 8, dtype=torch.int64, device=f"cuda:{dev}")
    h = v * _M1 + ((seed * 0x632BE59BD9B4E019) & ((1 << 62) - 1))
    h = h ^ (h >> 29)
    h = h * _M2
    h = h ^ (h >> 32)
    return h.view(torch.uint8)


class _Raw:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {
            "shape": (n,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def _fabric_prop(cu, dev):
    """A pinned device allocation on ``dev`` exportable as a FABRIC handle."""
    loc = cu.CUmemLocation()
    loc.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    loc.id = dev
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = loc
    prop.requestedHandleTypes = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    return loc, prop


def _vmm_pool(cu, dev, nchunks, hole=None):
    loc, prop = _fabric_prop(cu, dev)
    err, va = cu.cuMemAddressReserve(nchunks * CH, CH, 0, 0)
    assert int(err) == 0, err
    va = int(va)
    for i in range(nchunks):
        if i == hole:
            continue
        err, h = cu.cuMemCreate(CH, prop, 0)
        assert int(err) == 0, err
        (err,) = cu.cuMemMap(va + i * CH, CH, 0, h, 0)
        assert int(err) == 0, err
    acc = cu.CUmemAccessDesc()
    acc.location = loc
    acc.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    for lo, hi in [(0, nchunks)] if hole is None else [(0, hole), (hole + 1, nchunks)]:
        (err,) = cu.cuMemSetAccess(va + lo * CH, (hi - lo) * CH, [acc], 1)
        assert int(err) == 0, err
    return va, nchunks * CH


def owner_main():
    """The receiver ("decode") process: serves ZERO / CHECK / SAME commands on stdin."""
    import torch
    from cuda.bindings import driver as cu

    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
    from tensorrt_llm._torch.disaggregation.native import fabric_writer
    from tensorrt_llm._torch.disaggregation.native.transfer import _create_nixl_agent

    dev = OWNER_DEV
    torch.cuda.set_device(dev)
    torch.empty(1, device=f"cuda:{dev}")
    pa, pb = _vmm_pool(cu, dev, 24), _vmm_pool(cu, dev, 8)
    pc = _vmm_pool(cu, dev, 8, hole=3)  # exported with an unmapped chunk, not NIXL-registered
    other = torch.zeros(64 << 20, dtype=torch.uint8, device=f"cuda:{dev}")  # not exported
    regions = [(b, s, torch.as_tensor(_Raw(b, s), device=f"cuda:{dev}")) for b, s in (pa, pb)]
    regions.append((other.data_ptr(), other.numel(), other))
    for b, s in ((pc[0], 3 * CH), (pc[0] + 4 * CH, 4 * CH)):  # both sides of the hole
        regions.append((b, s, torch.as_tensor(_Raw(b, s), device=f"cuda:{dev}")))
    for _, _, t in regions:
        t.zero_()
    torch.cuda.synchronize(dev)
    # Registered regions need not be chunk aligned: pool a ends 3 MiB short of its last chunk and
    # pool b starts 5 MiB into its first one; the covering chunks are exported whole.
    blob = fabric_writer.export_pools(
        [
            (pa[0], pa[1] - (3 << 20), dev, "a"),
            (pb[0] + (5 << 20), pb[1] - (5 << 20), dev, "b"),
            (pc[0], pc[1], dev, "c"),
        ],
        dev,
    )
    assert blob, "export_pools returned nothing"
    agent = _create_nixl_agent(PEER + "0", rank=0, world_size=1)
    agent.register_memory(
        RegMemoryDescs("VRAM", [(b, s, dev, f"r{i}") for i, (b, s, _) in enumerate(regions[:3])])
    )
    desc = bytes(agent.get_local_agent_desc())

    def view(addr, size):
        for b, s, t in regions:
            if b <= addr and addr + size <= b + s:
                return t[addr - b : addr - b + size]
        raise ValueError(f"range {addr:#x}+{size} outside every region")

    out = {"regions": [[b, s] for b, s, _ in regions], "blob": blob.hex(), "desc": desc.hex()}
    print("READY " + json.dumps(out), flush=True)
    for line in sys.stdin:
        cmd, _, arg = line.strip().partition(" ")
        a = json.loads(arg) if arg else None
        if cmd == "STOP":
            break
        if cmd == "ZERO":
            for _, _, t in regions:
                t.zero_()
            torch.cuda.synchronize(dev)
            reply = {}
        elif cmd == "CHECK":
            bad_bytes = bad_frags = 0
            scratch = [t.clone() for _, _, t in regions]
            for addr, size, seed in a["frags"]:
                n = int((view(addr, size) != gen(torch, seed, size, dev)).sum().item())
                bad_bytes += n
                bad_frags += int(n > 0)
                for (b, s, _), sc in zip(regions, scratch):
                    if b <= addr < b + s:
                        sc[addr - b : addr - b + size].zero_()
            nz = sum(int(torch.count_nonzero(sc).item()) for sc in scratch)
            reply = {"bad_bytes": bad_bytes, "bad_frags": bad_frags, "nonzero_outside": nz}
        elif cmd == "SAME":
            diff = sum(int((view(x, n) != view(y, n)).sum().item()) for x, y, n in a["pairs"])
            reply = {"diff_bytes": diff}
        else:
            raise RuntimeError(f"unknown command {cmd}")
        print("OK " + json.dumps(reply), flush=True)
    agent.shutdown()


class _Owner:
    """The sender's handle on the receiver process."""

    def __init__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__), "owner"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            # The receiver exports its pools only when fabric writes are opted in.
            env={**os.environ, "TRTLLM_KV_FABRIC_WRITE": "1"},
        )
        line = self._reply("READY ", timeout=900)
        self.info = json.loads(line)

    def _reply(self, prefix, timeout=300):
        while True:
            box = []
            t = threading.Thread(target=lambda: box.append(self.p.stdout.readline()), daemon=True)
            t.start()
            t.join(timeout)
            if not box or not box[0]:
                raise RuntimeError(f"owner died or timed out (rc={self.p.poll()})")
            line = box[0].rstrip("\n")
            if line.startswith(prefix):
                return line[len(prefix) :]
            print("OWNER| " + line, flush=True)  # receiver logs are echoed

    def cmd(self, name, arg=None):
        self.p.stdin.write(name + (" " + json.dumps(arg) if arg is not None else "") + "\n")
        self.p.stdin.flush()
        return json.loads(self._reply("OK "))

    def stop(self):
        if self.p.poll() is not None:
            return self.p.returncode
        try:
            self.p.stdin.write("STOP\n")
            self.p.stdin.flush()
            self.p.wait(timeout=120)
        except Exception:
            self.p.kill()
        return self.p.returncode


class _Ctx:
    """Sender-side source arena; every destination is written at most once per test."""

    def __init__(self, torch, owner):
        self.torch = torch
        regs = [tuple(r) for r in owner.info["regions"]]
        self.pools, self.other, self.hole_regions = regs[:2], regs[2], regs[3:]
        self.src = torch.empty(2 << 30, dtype=torch.uint8, device=f"cuda:{SENDER_DEV}")
        self.seed = 1000
        self.rid = 1
        self.reset_src()

    def reset_src(self):
        self.src_off = 0
        self.used = set()

    def dst_slots(self, fsize):
        return [b + i * fsize for b, size in self.pools for i in range(size // fsize)]

    def make_request(self, dsts, fsize, strided):
        """Fill sender-side payloads; returns (frags [(dst, size, seed)], src ptrs)."""
        frags, srcs = [], []
        step = 2 * fsize if strided else fsize
        if self.src_off + step * len(dsts) > self.src.numel():
            raise RuntimeError("source arena exhausted")
        for d in dsts:
            assert int(d) not in self.used, "destination reused within a test"
            self.used.add(int(d))
            self.seed += 1
            off = self.src_off
            self.src[off : off + fsize].copy_(gen(self.torch, self.seed, fsize, SENDER_DEV))
            frags.append((int(d), fsize, self.seed))
            srcs.append(self.src.data_ptr() + off)
            self.src_off += step
        return frags, srcs


def _write_meta(ctx, frags, srcs):
    import numpy as np

    from tensorrt_llm._torch.disaggregation.native.transfer import WriteMeta

    ctx.rid += 1
    return WriteMeta(
        task=None,
        expected_transfers=1,
        peer_name=PEER + "0",
        peer_rank=0,
        peer_endpoint="",
        unique_rid=ctx.rid,
        src_ptrs=np.asarray(srcs, dtype=np.int64),
        dst_ptrs=np.asarray([f[0] for f in frags], dtype=np.int64),
        sizes=np.asarray([f[1] for f in frags], dtype=np.int64),
        dst_device_id=OWNER_DEV,
        slice_id=0,
    )


def _send(sender, ctx, frags, srcs):
    """Submit one request; returns (route, ok, detail) with route fabric, nixl or fabric+fallback."""
    from tensorrt_llm._torch.disaggregation.native import fabric_writer

    wm = _write_meta(ctx, frags, srcs)
    # A request without physical-ownership enforcement: the only kind fabric writes serve.
    request = sender._build_kv_request(wm, False)
    route = "fabric" if fabric_writer.is_fabric_job(request) else "nixl"
    before = sender._fabric.stats().get("fallbacks", 0)
    task = SimpleNamespace(
        _enforce_physical_ownership=False,
        _unique_rid=wm.unique_rid,
        lock=threading.Lock(),
        _native_inflight=0,
    )
    ok, detail = sender._submit_transfer(task, 0, request)
    if route == "fabric" and sender._fabric.stats().get("fallbacks", 0) != before:
        route = "fabric+fallback"  # the copy failed and NIXL re-issued the request
    return route, ok, detail


def _new_sender(agent, owner, env=None):
    """A fabric-enabled Sender registered with the receiver (its import ready); ``env`` adds fabric knobs."""
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    from tensorrt_llm._torch.disaggregation.native.transfer import MessageType, Sender

    for k in [k for k in os.environ if k.startswith("TRTLLM_KV_FABRIC_WRITE")]:
        del os.environ[k]
    os.environ["TRTLLM_KV_FABRIC_WRITE"] = "1"
    os.environ.update(env or {})
    reg = SimpleNamespace(
        self_rank_info=SimpleNamespace(
            device_id=SENDER_DEV, instance_rank=0, instance_name="ut_pre"
        ),
        register=lambda *a, **k: None,
    )
    sender = Sender(reg, agent)
    assert sender._fabric is not None
    ri = RankInfo(
        instance_name=PEER,
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[1],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=bytes.fromhex(owner.info["desc"]),
        device_id=OWNER_DEV,
        fabric_export=bytes.fromhex(owner.info["blob"]),
    )
    sender._register_peer_rank(b"", [MessageType.REGISTER_RANK_INFO, ri.to_bytes()])
    t0 = time.time()
    while not sender._fabric.peer_ready(PEER + "0"):
        if time.time() - t0 > 120:
            raise RuntimeError("peer import did not become ready")
        time.sleep(0.05)
    return sender


def _inject_issue_fault(writer, kind):
    """Fail one batch issue of the next request ("first": its first batch, "later": a later one)."""
    fw = sys.modules[type(writer).__module__]
    issue = writer._issue
    left = [1]

    def faulty_issue(job, a, b, stream, event):
        if left[0] > 0 and ((kind == "first" and a == 0) or (kind == "later" and a > 0)):
            left[0] -= 1
            raise fw._IssueError(f"injected test fault ({kind}) at op {a}", enqueued=False)
        return issue(job, a, b, stream, event)

    writer._issue = faulty_issue


def _check(owner, frags):
    r = owner.cmd("CHECK", {"frags": frags})
    assert r["bad_bytes"] == 0 and r["bad_frags"] == 0 and r["nonzero_outside"] == 0, r


def sender_main():
    """The sender ("prefill") process: runs every case and prints ALL_PASS."""
    import msgpack
    import torch

    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
    from tensorrt_llm._torch.disaggregation.native.transfer import Sender, _create_nixl_agent

    torch.cuda.set_device(SENDER_DEV)
    torch.empty(1, device=f"cuda:{SENDER_DEV}")
    rng = random.Random(1234)
    owner = _Owner()
    try:
        ctx = _Ctx(torch, owner)
        agent = _create_nixl_agent("ut_pre0", rank=0, world_size=1)
        agent.register_memory(
            RegMemoryDescs("VRAM", [(ctx.src.data_ptr(), ctx.src.numel(), SENDER_DEV, "src")])
        )

        def fresh(fsize, n):
            slots = [x for x in ctx.dst_slots(fsize) if x not in ctx.used]
            return ctx.make_request(rng.sample(slots, n), fsize, strided=True)

        def t_fbw(s):
            """4 concurrent producers, distinct payloads and destinations."""
            fsize, nfr = 128 << 10, 600
            # Six contiguous runs in pool a (crossing 32 MiB chunk boundaries, contiguous
            # sources), then six random strided requests.
            pool_a = ctx.pools[0]
            runs = [pool_a[0] + k * fsize for k in range(pool_a[1] // fsize)]
            reqs = [ctx.make_request(runs[i * nfr : (i + 1) * nfr], fsize, False) for i in range(6)]
            reqs += [fresh(fsize, nfr) for _ in range(6)]
            torch.cuda.synchronize()
            out = [None] * len(reqs)

            def producer(tid):
                for k in range(tid, len(reqs), 4):
                    out[k] = _send(s, ctx, *reqs[k])

            th = [threading.Thread(target=producer, args=(t,)) for t in range(4)]
            for t in th:
                t.start()
            for t in th:
                t.join()
            assert all(o is not None and o[0] == "fabric" and o[1] for o in out), out
            _check(owner, [f for r in reqs for f in r[0]])
            st = s._fabric.stats()
            assert st.get("requests") == 12 and st.get("fallbacks", 0) == 0, st
            assert st.get("copy_errors", 0) == 0, st
            assert st["ops"] < st["descs"], st  # contiguous requests coalesced
            assert st["batches"] >= 12, st
            nch = len(msgpack.unpackb(bytes.fromhex(owner.info["blob"]))["chunks"])
            assert nch == 39, f"export walked {nch} chunks, expected 39 (24 + 8 + 7 x 32 MiB)"

        def t_same(s):
            """FBW and NIXL write the same payloads to two regions: both exact and identical."""
            fsize, n = 256 << 10, 300
            slots = ctx.dst_slots(fsize)
            rng.shuffle(slots)
            xa, ya = slots[:n], slots[n : 2 * n]
            fa, sa = ctx.make_request(xa, fsize, strided=True)
            fb = [(y, fsize, f[2]) for y, f in zip(ya, fa)]  # the same payloads, via NIXL
            torch.cuda.synchronize()
            ra = _send(s, ctx, fa, sa)
            assert ra[0] == "fabric" and ra[1], ra
            req = Sender._make_agent_request(_write_meta(ctx, fb, sa), SENDER_DEV)
            assert agent.submit_transfer_requests(req).wait(), "NIXL write failed"
            _check(owner, fa + fb)
            d = owner.cmd("SAME", {"pairs": [[x, y, fsize] for x, y in zip(xa, ya)]})
            assert d["diff_bytes"] == 0, d

        def t_gate(s):
            """Small fragments and unexported destinations route to NIXL; 64 KiB goes fabric."""
            f1, s1 = fresh(16 << 10, 64)
            ob, osz = ctx.other
            oslots = [ob + i * (128 << 10) for i in range(osz // (128 << 10))]
            f2, s2 = ctx.make_request(rng.sample(oslots, 50), 128 << 10, strided=True)
            f3, s3 = fresh(63 << 10, 32)  # just under the gate
            f4, s4 = fresh(64 << 10, 32)  # at the gate
            torch.cuda.synchronize()
            routes = [_send(s, ctx, f, sp) for f, sp in ((f1, s1), (f2, s2), (f3, s3), (f4, s4))]
            assert [r[0] for r in routes] == ["nixl", "nixl", "nixl", "fabric"], routes
            assert all(r[1] for r in routes), routes
            st = s._fabric.stats()
            assert st.get("inelig_small_fragments") == 2, st
            assert st.get("inelig_dst_unmapped") == 1, st
            _check(owner, f1 + f2 + f3 + f4)

        def t_hole(s):
            """A pool exported around an unmapped chunk: both sides of the hole are writable."""
            fsize = 256 << 10
            slots = [b + i * fsize for b, sz in ctx.hole_regions for i in range(sz // fsize)]
            f, sp = ctx.make_request(rng.sample(slots, 120), fsize, strided=True)
            torch.cuda.synchronize()
            r = _send(s, ctx, f, sp)
            assert r[0] == "fabric" and r[1], r
            _check(owner, f)

        def t_fault(s, kind):
            """A failed batch re-issues the whole request through NIXL, then disables the peer; a later
            batch's failure first drains the batches already enqueued."""
            _inject_issue_fault(s._fabric, kind)
            f1, s1 = fresh(128 << 10, 100)
            f2, s2 = fresh(128 << 10, 40)
            torch.cuda.synchronize()
            r1 = _send(s, ctx, f1, s1)
            r2 = _send(s, ctx, f2, s2)
            st = s._fabric.stats()
            assert r1[0] == "fabric+fallback" and r1[1], r1
            assert st.get("fallbacks") == 1 and st.get("copy_errors") == 1, st
            assert r2[0] == "nixl" and r2[1], r2
            assert st.get("inelig_peer_disabled") == 1, st
            _check(owner, f1 + f2)

        results = {}
        cases = (
            (t_fbw, None),
            (t_same, None),
            (t_gate, None),
            (t_hole, None),
            (lambda s: t_fault(s, "first"), None),
            # Eight ops per batch: the 100-fragment request spans several batches.
            (lambda s: t_fault(s, "later"), {"TRTLLM_KV_FABRIC_WRITE_OPS_PER_BATCH": "8"}),
        )
        for name, (fn, env) in zip(EXPECTED_TESTS, cases):
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, owner, env)
            try:
                fn(s)
                results[name], detail = True, ""
            except Exception:
                results[name], detail = False, traceback.format_exc().replace("\n", " | ")
            finally:
                s.shutdown()
            print(f"UT_RESULT {name} {'PASS' if results[name] else 'FAIL'} {detail}", flush=True)
        rc = owner.stop()
        agent.shutdown()
        ok = all(results.values()) and len(results) == len(EXPECTED_TESTS) and rc == 0
        print("ALL_PASS" if ok else f"UT_FAIL owner_rc={rc}", flush=True)
        return 0 if ok else 1
    finally:
        # Stop and reap the receiver even when the sender fails before its own stop.
        owner.stop()


# Bound on the whole two-process run (the receiver alone may take minutes to build its pools).
_RUN_TIMEOUT_S = 1800


def _fabric_supported(cu, dev: int) -> bool:
    """Whether ``dev`` can create and export a FABRIC allocation: the attribute alone is not enough
    on a node without an IMEX channel, where cuMemCreate is not permitted."""
    err, supported = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev
    )
    if err != cu.CUresult.CUDA_SUCCESS or not supported:
        return False
    err, ctx = cu.cuDevicePrimaryCtxRetain(dev)
    if err != cu.CUresult.CUDA_SUCCESS:
        return False
    try:
        (err,) = cu.cuCtxPushCurrent(ctx)
        if err != cu.CUresult.CUDA_SUCCESS:
            return False
        try:
            _, prop = _fabric_prop(cu, dev)
            err, gran = cu.cuMemGetAllocationGranularity(
                prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            )
            if err != cu.CUresult.CUDA_SUCCESS:
                return False
            err, handle = cu.cuMemCreate(gran, prop, 0)
            if err != cu.CUresult.CUDA_SUCCESS:
                return False
            try:
                err, _ = cu.cuMemExportToShareableHandle(
                    handle, cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC, 0
                )
                return err == cu.CUresult.CUDA_SUCCESS
            finally:
                cu.cuMemRelease(handle)
        finally:
            cu.cuCtxPopCurrent()
    finally:
        cu.cuDevicePrimaryCtxRelease(dev)


def test_fabric_writer_2gpu():
    import torch

    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2 GPUs with FABRIC handle support")
    pytest.importorskip("cuda.bindings")
    from cuda.bindings import driver as cu

    (err,) = cu.cuInit(0)
    assert err == cu.CUresult.CUDA_SUCCESS, err
    for dev in (SENDER_DEV, OWNER_DEV):
        if not _fabric_supported(cu, dev):
            pytest.skip(f"cuda:{dev}: FABRIC cuMemCreate/export not permitted (no IMEX channel?)")
    try:
        from tensorrt_llm._torch.disaggregation.nixl import agent as nixl_agent_mod
    except ImportError as e:
        pytest.skip(f"NIXL transfer agent unavailable: {e}")
    if nixl_agent_mod.NixlTransferAgent is None:
        pytest.skip("NIXL transfer agent unavailable")
    # Own session, so a run past the deadline is killed together with its receiver child.
    p = subprocess.Popen(
        [sys.executable, "-u", os.path.abspath(__file__)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = p.communicate(timeout=_RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        pytest.fail(f"timed out after {_RUN_TIMEOUT_S} s: " + out[-4000:] + err[-4000:])
    assert p.returncode == 0 and "ALL_PASS" in out, out[-4000:] + err[-4000:]


if __name__ == "__main__":
    # ``^ib,gdr_copy`` as in the other disaggregated transfer tests: GDR copy cannot pin the VMM
    # pools both processes register, and InfiniBand is not needed between two local GPUs. Set in
    # the sender and owner processes only; the owner also inherits it from the sender.
    os.environ["UCX_TLS"] = "^ib,gdr_copy"
    if len(sys.argv) > 1 and sys.argv[1] == "owner":
        owner_main()
        sys.exit(0)
    sys.exit(sender_main())
