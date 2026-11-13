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
process drives a real ``Sender`` (routing, backend submission, NIXL fallback). Every written
fragment carries its own payload (unique seed) and destination; the receiver verifies each one
byte by byte against an independently regenerated payload and checks every other byte is zero.

Run directly: ``python3 test_fabric_writer_2gpu.py`` (prints UT_RESULT lines and ALL_PASS).
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
    "knob0_identical_routing",
    "fbw_concurrent_byte_exact",
    "fbw_equals_nixl",
    "gate_routing",
    "export_with_hole",
    "refuse_no_export",
    "refuse_sleep_rebalance",
    "fault_first_batch",
    "fault_later_batch",
    "negative_controls",
]

_M1 = -7046029254386353131  # 0x9E3779B97F4A7C15 as int64
_M2 = -4658895280553007687  # 0xBF58476D1CE4E5B9 as int64


def gen(torch, seed: int, nbytes: int, dev: int):
    """Deterministic integer-hash payload (identical on any GPU), ``nbytes`` multiple of 8."""
    assert nbytes % 8 == 0
    v = torch.arange(nbytes // 8, dtype=torch.int64, device=f"cuda:{dev}")
    h = v * _M1 + ((seed * 0x632BE59BD9B4E019) & ((1 << 62) - 1))
    h = h ^ (h >> 29)
    h = h * _M2
    h = h ^ (h >> 32)
    return h.view(torch.uint8)


# ------------------------------------------------------------------------------------------------
# receiver ("decode") process
# ------------------------------------------------------------------------------------------------
class _Raw:
    def __init__(self, ptr, n):
        self.__cuda_array_interface__ = {
            "shape": (n,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def _vmm_pool(cu, dev, nchunks, hole=None):
    fab = cu.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    loc = cu.CUmemLocation()
    loc.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    loc.id = dev
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location = loc
    prop.requestedHandleTypes = fab
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
    import torch
    from cuda.bindings import driver as cu

    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
    from tensorrt_llm._torch.disaggregation.native import fabric_writer
    from tensorrt_llm._torch.disaggregation.native.transfer import _create_nixl_agent

    dev = OWNER_DEV
    torch.cuda.set_device(dev)
    torch.empty(1, device=f"cuda:{dev}")
    pa = _vmm_pool(cu, dev, 24)
    pb = _vmm_pool(cu, dev, 8)
    pc = _vmm_pool(cu, dev, 8, hole=3)  # exported with an unmapped chunk, not NIXL-registered
    other = torch.zeros(64 << 20, dtype=torch.uint8, device=f"cuda:{dev}")
    c1 = (pc[0], 3 * CH)
    c2 = (pc[0] + 4 * CH, 4 * CH)
    regions = [
        (pa[0], pa[1], torch.as_tensor(_Raw(*pa), device=f"cuda:{dev}")),
        (pb[0], pb[1], torch.as_tensor(_Raw(*pb), device=f"cuda:{dev}")),
        (other.data_ptr(), other.numel(), other),
        (c1[0], c1[1], torch.as_tensor(_Raw(*c1), device=f"cuda:{dev}")),
        (c2[0], c2[1], torch.as_tensor(_Raw(*c2), device=f"cuda:{dev}")),
    ]
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

    out = {
        "regions": [[b, s] for b, s, _ in regions],
        "blob": blob.hex(),
        "desc": desc.hex(),
    }
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
            print("OK {}", flush=True)
        elif cmd == "CHECK":
            bad_bytes = bad_frags = 0
            scratch = [t.clone() for _, _, t in regions]
            for addr, size, seed in a["frags"]:
                got = view(addr, size)
                n = int((got != gen(torch, seed, size, dev)).sum().item())
                bad_bytes += n
                bad_frags += int(n > 0)
                for (b, s, _), sc in zip(regions, scratch):
                    if b <= addr < b + s:
                        sc[addr - b : addr - b + size].zero_()
            nz = sum(int(torch.count_nonzero(sc).item()) for sc in scratch)
            print(
                "OK "
                + json.dumps(
                    {"bad_bytes": bad_bytes, "bad_frags": bad_frags, "nonzero_outside": nz}
                ),
                flush=True,
            )
        elif cmd == "SAME":
            diff = 0
            for x, y, size in a["pairs"]:
                diff += int((view(x, size) != view(y, size)).sum().item())
            print("OK " + json.dumps({"diff_bytes": diff}), flush=True)
        elif cmd == "PERMUTE":
            w = view(a["addr"], a["size"]).view(torch.int32)
            j = int(torch.nonzero(w != w[0])[0].item())
            w0, wj = w[0].clone(), w[j].clone()
            w[0], w[j] = wj, w0
            torch.cuda.synchronize(dev)
            print("OK " + json.dumps({"swapped": [0, j]}), flush=True)
        elif cmd == "FLIP":
            v = view(a["addr"], 1)
            v.bitwise_xor_(torch.ones(1, dtype=torch.uint8, device=f"cuda:{dev}"))
            torch.cuda.synchronize(dev)
            print("OK {}", flush=True)
        elif cmd == "POKE":
            view(a["addr"], 1).fill_(7)
            torch.cuda.synchronize(dev)
            print("OK {}", flush=True)
        else:
            raise RuntimeError(f"unknown command {cmd}")
    agent.shutdown()


# RankInfo's wire fields before the fabric export and the drafter window were added, in field order. A
# rank without either must serialize exactly these, so a peer built before them parses its message.
_LEGACY_RANK_INFO_FIELDS = (
    "instance_name",
    "instance_rank",
    "tp_size",
    "tp_rank",
    "pp_size",
    "pp_rank",
    "layer_num_per_pp",
    "sender_endpoints",
    "self_endpoint",
    "transfer_engine_info",
    "dp_size",
    "dp_rank",
    "cp_size",
    "cp_rank",
    "device_id",
    "attention",
    "aux_meta",
    "page_table",
    "bounded_replay_on_generation",
    "protocol_version",
    "context_first_ownership",
    "num_draft_layers",
)


def _rank_info_keeps_its_legacy_bytes() -> bool:
    """RankInfo.to_bytes of a rank without an export equals the legacy serialization of its fields."""
    from dataclasses import asdict

    import msgpack

    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

    info = RankInfo(
        instance_name="x",
        instance_rank=3,
        tp_size=8,
        tp_rank=3,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[4],
        sender_endpoints=["tcp://a:1"],
        self_endpoint="tcp://b:2",
        transfer_engine_info=b"\x01\x02",
        device_id=3,
    )
    fields = asdict(info)
    legacy = {name: fields[name] for name in _LEGACY_RANK_INFO_FIELDS}
    # The legacy encoder drops context_first_ownership when unset; attention, aux_meta and
    # page_table are unset here.
    legacy.pop("context_first_ownership")
    exported = RankInfo(**{**fields, "fabric_export": b"\x07"}).to_bytes()
    return info.to_bytes() == msgpack.packb(legacy) and info.to_bytes() != exported


def test_rank_info_without_an_export_keeps_its_legacy_bytes():
    assert _rank_info_keeps_its_legacy_bytes()


# ------------------------------------------------------------------------------------------------
# sender ("prefill") process
# ------------------------------------------------------------------------------------------------
class _Owner:
    def __init__(self):
        self.p = subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__), "owner"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            # The receiver exports its pools only when fabric writes are opted in.
            env={**os.environ, "TRTLLM_KV_FABRIC_WRITE": "1"},
        )
        while True:
            line = self._line(timeout=900)
            if line.startswith("READY "):
                break
            print("OWNER| " + line, flush=True)
        self.info = json.loads(line[6:])

    def _line(self, timeout=300):
        box = []
        t = threading.Thread(target=lambda: box.append(self.p.stdout.readline()), daemon=True)
        t.start()
        t.join(timeout)
        if not box or not box[0]:
            raise RuntimeError(f"owner died or timed out (rc={self.p.poll()})")
        return box[0].rstrip("\n")

    def cmd(self, name, arg=None):
        self.p.stdin.write(name + (" " + json.dumps(arg) if arg is not None else "") + "\n")
        self.p.stdin.flush()
        while True:
            line = self._line()
            if line.startswith("OK "):
                return json.loads(line[3:])
            if line.startswith("READY "):
                continue
            # anything else from the owner (logs) is echoed
            print("OWNER| " + line, flush=True)

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
    def __init__(self, torch, owner):
        self.torch = torch
        self.owner = owner
        regs = [tuple(r) for r in owner.info["regions"]]
        self.pools = regs[:2]
        self.other = regs[2]
        self.hole_regions = regs[3:]
        self.src = torch.empty(2 << 30, dtype=torch.uint8, device=f"cuda:{SENDER_DEV}")
        self.src_off = 0
        self.seed = 1000
        self.rid = 1
        self.used = set()

    def next_seed(self):
        self.seed += 1
        return self.seed

    def reset_src(self):
        self.src_off = 0
        self.used = set()

    def dst_slots(self, fsize):
        slots = []
        for base, size in self.pools:
            slots.extend(base + i * fsize for i in range(size // fsize))
        return slots

    def make_request(self, dsts, fsize, strided):
        """Fill sender-side payloads; returns (frags [(dst, size, seed)], src ptrs)."""
        torch = self.torch
        frags, srcs = [], []
        step = 2 * fsize if strided else fsize
        if self.src_off + step * len(dsts) > self.src.numel():
            raise RuntimeError("source arena exhausted")
        for d in dsts:
            assert int(d) not in self.used, "destination reused within a test"
            self.used.add(int(d))
            seed = self.next_seed()
            off = self.src_off
            self.src[off : off + fsize].copy_(gen(torch, seed, fsize, SENDER_DEV))
            frags.append((int(d), fsize, seed))
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
    from tensorrt_llm._torch.disaggregation.native import fabric_writer

    wm = _write_meta(ctx, frags, srcs)
    # A request without physical-ownership enforcement: the only kind fabric writes serve.
    request = sender._build_kv_request(wm, False)
    route = "fabric" if fabric_writer.is_fabric_job(request) else "nixl"
    before = sender._fabric.stats().get("fallbacks", 0) if sender._fabric is not None else 0
    task = SimpleNamespace(_enforce_physical_ownership=False, _unique_rid=wm.unique_rid)
    ok, detail = sender._submit_transfer(task, 0, request)
    after = sender._fabric.stats().get("fallbacks", 0) if sender._fabric is not None else 0
    if route == "fabric" and after != before:
        route = "fabric+fallback"  # the copy failed and NIXL re-issued the request
    return route, ok, detail


def _new_sender(agent, env):
    from tensorrt_llm._torch.disaggregation.native.transfer import Sender

    for k in [k for k in os.environ if k.startswith("TRTLLM_KV_FABRIC_WRITE")]:
        del os.environ[k]
    os.environ.update(env)
    reg = SimpleNamespace(
        self_rank_info=SimpleNamespace(
            device_id=SENDER_DEV, instance_rank=0, instance_name="ut_pre"
        ),
        register=lambda *a, **k: None,
    )
    return Sender(reg, agent)


def _register(sender, owner, with_export=True):
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
    from tensorrt_llm._torch.disaggregation.native.transfer import MessageType

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
        fabric_export=bytes.fromhex(owner.info["blob"]) if with_export else None,
    )
    sender._register_peer_rank(b"", [MessageType.REGISTER_RANK_INFO, ri.to_bytes()])
    if with_export and sender._fabric is not None:
        t0 = time.time()
        while not sender._fabric.peer_ready(PEER + "0"):
            if time.time() - t0 > 120:
                raise RuntimeError("peer import did not become ready")
            time.sleep(0.05)


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
    return r["bad_bytes"] == 0 and r["bad_frags"] == 0 and r["nonzero_outside"] == 0, r


def sender_main():
    import torch

    from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
    from tensorrt_llm._torch.disaggregation.native import fabric_writer
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
        results = {}

        def record(name, ok, detail):
            results[name] = bool(ok)
            print(f"UT_RESULT {name} {'PASS' if ok else 'FAIL'} {detail}", flush=True)

        def run(name, fn):
            try:
                ok, detail = fn()
            except Exception:
                ok, detail = False, traceback.format_exc().replace("\n", " | ")
            record(name, ok, detail)

        def fresh(fsize, n, strided=False, contiguous=False, pool_slots=None):
            slots = pool_slots if pool_slots is not None else ctx.dst_slots(fsize)
            slots = [x for x in slots if x not in ctx.used]
            if contiguous:
                start = rng.randrange(0, len(slots) - n)
                dsts = slots[start : start + n]
                # only keep it contiguous if the run stays inside one pool
                if dsts[-1] - dsts[0] != (n - 1) * fsize:
                    dsts = slots[:n]
            else:
                dsts = rng.sample(slots, n)
            return ctx.make_request(dsts, fsize, strided)

        # ---- 1. knob 0: transfer-agent routing exactly as before, RankInfo bytes unchanged ----
        def t_knob0():
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, {"TRTLLM_KV_FABRIC_WRITE": "0"})
            try:
                assert s._fabric is None, "writer created with knob 0"
                _register(s, owner)
                frags, srcs = fresh(128 << 10, 200, strided=True)
                route, ok, detail = _send(s, ctx, frags, srcs)
                assert route == "nixl" and ok, (route, ok, detail)
                good, r = _check(owner, frags)
                assert good, r
                assert fabric_writer.export_pools([(1, 1, 0, "x")], 0) is None
            finally:
                s.shutdown()
            assert _rank_info_keeps_its_legacy_bytes(), "RankInfo bytes changed without an export"
            return True, f"route=nixl byte-exact frags={len(frags)} rankinfo_bytes_identical=1"

        # ---- 2. FBW: 4 concurrent producers, distinct payloads and destinations ----
        def t_fbw():
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, {"TRTLLM_KV_FABRIC_WRITE": "1"})
            try:
                assert s._fabric is not None
                _register(s, owner)
                fsize, nfr = 128 << 10, 600
                slots = ctx.dst_slots(fsize)
                rng.shuffle(slots)
                reqs = []
                for i in range(12):
                    if i < 6:
                        # contiguous destination run (crosses 32 MiB chunk boundaries) + contiguous source
                        dsts = None
                        for b, sz in ctx.pools:
                            cand = [b + k * fsize for k in range(sz // fsize)]
                            free = sorted(set(cand) - set(x for r in reqs for x in r[2]))
                            for st in range(0, len(free) - nfr):
                                if free[st + nfr - 1] - free[st] == (nfr - 1) * fsize:
                                    dsts = free[st : st + nfr]
                                    break
                            if dsts:
                                break
                        assert dsts, "no contiguous run left"
                        frags, srcs = ctx.make_request(dsts, fsize, strided=False)
                    else:
                        taken = set(x for r in reqs for x in r[2])
                        cand = [x for x in slots if x not in taken]
                        dsts = rng.sample(cand, nfr)
                        frags, srcs = ctx.make_request(dsts, fsize, strided=True)
                    reqs.append((frags, srcs, dsts))
                torch.cuda.synchronize()
                all_dst = [x for r in reqs for x in r[2]]
                assert len(all_dst) == len(set(all_dst)), "destinations overlap"
                out = [None] * len(reqs)

                def producer(tid):
                    for k in range(tid, len(reqs), 4):
                        out[k] = _send(s, ctx, reqs[k][0], reqs[k][1])

                th = [threading.Thread(target=producer, args=(t,)) for t in range(4)]
                for t in th:
                    t.start()
                for t in th:
                    t.join()
                assert all(o is not None for o in out), out
                routes = [o[0] for o in out]
                assert all(o[0] == "fabric" and o[1] for o in out), out
                frags = [f for r in reqs for f in r[0]]
                good, r = _check(owner, frags)
                st = s._fabric.stats()
                assert good, r
                assert st.get("requests") == 12 and st.get("fallbacks", 0) == 0, st
                assert st.get("copy_errors", 0) == 0, st
                assert st["ops"] < st["descs"], st  # contiguous requests coalesced
                assert st["batches"] >= 12, st
                ctx.t2_frags = frags
                import msgpack

                nch = len(msgpack.unpackb(bytes.fromhex(owner.info["blob"]))["chunks"])
                assert nch == 39, f"export walked {nch} chunks, expected 39 (24 + 8 + 7 x 32 MiB)"
                return True, (
                    f"export_chunks={nch} "
                    + f"routes={routes.count('fabric')}/12 fabric frags={len(frags)} bad_bytes=0 "
                    f"nonzero_outside=0 descs={st['descs']} ops={st['ops']} batches={st['batches']} "
                    f"lat_ms_avg={st['lat_ms_sum'] / st['requests']:.2f}"
                )
            finally:
                s.shutdown()

        # ---- 3. FBW and NIXL write the same payloads to two regions: both exact and identical ----
        def t_same():
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, {"TRTLLM_KV_FABRIC_WRITE": "1"})
            try:
                _register(s, owner)
                fsize, n = 256 << 10, 300
                slots = ctx.dst_slots(fsize)
                rng.shuffle(slots)
                xa, ya = slots[:n], slots[n : 2 * n]
                fa, sa = ctx.make_request(xa, fsize, strided=True)
                # same seeds for the NIXL copy: rebuild sources with identical payloads
                fb = [(y, fsize, f[2]) for y, f in zip(ya, fa)]
                torch.cuda.synchronize()
                ra = _send(s, ctx, fa, sa)
                assert ra[0] == "fabric" and ra[1], ra
                req = Sender._make_agent_request(_write_meta(ctx, fb, sa), SENDER_DEV)
                st = agent.submit_transfer_requests(req)
                assert st.wait(), "NIXL write failed"
                good, r = _check(owner, fa + fb)
                assert good, r
                d = owner.cmd("SAME", {"pairs": [[x, y, fsize] for x, y in zip(xa, ya)]})
                assert d["diff_bytes"] == 0, d
                return True, f"fabric+nixl frags={2 * n} bad_bytes=0 fabric_vs_nixl_diff_bytes=0"
            finally:
                s.shutdown()

        # ---- 4. gate routing: small fragments and unexported destinations stay on NIXL ----
        def t_gate():
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, {"TRTLLM_KV_FABRIC_WRITE": "1"})
            try:
                _register(s, owner)
                f1, s1 = fresh(16 << 10, 64, strided=True)
                r1 = _send(s, ctx, f1, s1)
                ob, osz = ctx.other
                oslots = [ob + i * (128 << 10) for i in range(osz // (128 << 10))]
                f2, s2 = ctx.make_request(rng.sample(oslots, 50), 128 << 10, strided=True)
                r2 = _send(s, ctx, f2, s2)
                # 63 KiB average (just under the gate) and exactly 64 KiB (at the gate)
                f3, s3 = fresh(63 << 10 if (63 << 10) % 8 == 0 else 64512, 32, strided=True)
                r3 = _send(s, ctx, f3, s3)
                f4, s4 = fresh(64 << 10, 32, strided=True)
                r4 = _send(s, ctx, f4, s4)
                torch.cuda.synchronize()
                st = s._fabric.stats()
                assert r1[0] == "nixl" and r1[1], r1
                assert r2[0] == "nixl" and r2[1], r2
                assert r3[0] == "nixl" and r3[1], r3
                assert r4[0] == "fabric" and r4[1], r4
                assert st.get("inelig_small_fragments") == 2, st
                assert st.get("inelig_dst_unmapped") == 1, st
                good, r = _check(owner, f1 + f2 + f3 + f4)
                assert good, r
                return True, (
                    "16KiB->nixl unexported_dst->nixl 63KiB->nixl 64KiB->fabric all byte-exact "
                    f"inelig={ {k: v for k, v in st.items() if k.startswith('inelig_')} }"
                )
            finally:
                s.shutdown()

        # ---- 4b. a pool exported around an unmapped chunk: both sides of the hole are writable ----
        def t_hole():
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, {"TRTLLM_KV_FABRIC_WRITE": "1"})
            try:
                _register(s, owner)
                fsize = 256 << 10
                slots = []
                for b, sz in ctx.hole_regions:
                    slots.extend(b + i * fsize for i in range(sz // fsize))
                f, sp = ctx.make_request(rng.sample(slots, 120), fsize, strided=True)
                torch.cuda.synchronize()
                r = _send(s, ctx, f, sp)
                assert r[0] == "fabric" and r[1], r
                good, c = _check(owner, f)
                assert good, c
                return True, "pool with unmapped chunk: 120 frags on both sides fabric byte-exact"
            finally:
                s.shutdown()

        # ---- 5. a peer without an export (refused) stays on NIXL ----
        def t_noexport():
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, {"TRTLLM_KV_FABRIC_WRITE": "1"})
            try:
                _register(s, owner, with_export=False)
                f, sp = fresh(128 << 10, 100, strided=True)
                torch.cuda.synchronize()
                r = _send(s, ctx, f, sp)
                assert r[0] == "nixl" and r[1], r
                good, c = _check(owner, f)
                assert good, c
                assert s._fabric.stats().get("inelig_no_peer_export") == 1
                return True, "peer without export -> nixl byte-exact"
            finally:
                s.shutdown()

        # ---- 6. receiver-side refusal: sleep scope, explicit rebalance refusal, executor gate ----
        def t_refuse():
            code = (
                "import sys\n"
                "from tensorrt_llm._torch import virtual_memory as vm\n"
                "from tensorrt_llm._torch.disaggregation.native import fabric_writer as fw\n"
                "mode = sys.argv[1]\n"
                "if mode == 'sleep':\n"
                "    vm._pool_stack.append((None, None)); fw.note_worker_init(); vm._pool_stack.pop()\n"
                "else:\n"
                "    fw.refuse('kv pool rebalance enabled')\n"
                "r = fw.refusal_reason(); b = fw.export_pools([(1 << 40, 1 << 21, 0, 'x')], 0)\n"
                "print('REFUSAL', mode, repr(r), b is None)\n"
                "sys.exit(0 if (r and b is None) else 3)\n"
            )
            outs = []
            for mode in ("sleep", "rebalance"):
                p = subprocess.run(
                    [sys.executable, "-c", code, mode], capture_output=True, text=True, timeout=300
                )
                line = [x for x in p.stdout.splitlines() if x.startswith("REFUSAL")]
                assert p.returncode == 0 and line, (
                    p.returncode,
                    p.stdout[-2000:],
                    p.stderr[-2000:],
                )
                outs.append(line[0])
            # no refusal without a sleep scope
            p = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from tensorrt_llm._torch.disaggregation.native import fabric_writer as fw\n"
                    "fw.note_worker_init(); print('NOREFUSAL', fw.refusal_reason())",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            assert p.returncode == 0 and "NOREFUSAL None" in p.stdout, p.stdout[-2000:]
            import tensorrt_llm._torch.pyexecutor.py_executor as pe

            src = open(pe.__file__).read()
            gate = "if self.kv_cache_transceiver is not None:\n            return False"
            assert gate in src, "executor no longer blocks rebalance while a transceiver exists"
            return True, " ; ".join(outs) + " ; executor rebalance gate present"

        # ---- 7/8. injected batch faults: whole request re-issued through NIXL, peer then disabled ----
        def t_fault(kind):
            def body():
                owner.cmd("ZERO")
                ctx.reset_src()
                env = {"TRTLLM_KV_FABRIC_WRITE": "1"}
                if kind == "later":
                    env["TRTLLM_KV_FABRIC_WRITE_OPS_PER_BATCH"] = "8"
                s = _new_sender(agent, env)
                _inject_issue_fault(s._fabric, kind)
                try:
                    _register(s, owner)
                    f1, s1 = fresh(128 << 10, 100, strided=True)
                    f2, s2 = fresh(128 << 10, 40, strided=True)
                    torch.cuda.synchronize()
                    r1 = _send(s, ctx, f1, s1)
                    r2 = _send(s, ctx, f2, s2)
                    st = s._fabric.stats()
                    assert r1[0] == "fabric+fallback" and r1[1], r1
                    assert st.get("fallbacks") == 1 and st.get("copy_errors") == 1, st
                    assert r2[0] == "nixl" and r2[1], r2
                    assert st.get("inelig_peer_disabled") == 1, st
                    good, c = _check(owner, f1 + f2)
                    assert good, c
                    return (
                        True,
                        f"fault={kind} -> nixl fallback byte-exact, next request nixl; stats={st}",
                    )
                finally:
                    s.shutdown()

            return body

        # ---- 9. the checker catches a sum-preserving swap, a flip and a gap write ----
        def t_neg():
            owner.cmd("ZERO")
            ctx.reset_src()
            s = _new_sender(agent, {"TRTLLM_KV_FABRIC_WRITE": "1"})
            try:
                _register(s, owner)
                f, sp = fresh(128 << 10, 64, strided=True)
                torch.cuda.synchronize()
                r = _send(s, ctx, f, sp)
                assert r[0] == "fabric" and r[1], r
            finally:
                s.shutdown()
            good, c0 = _check(owner, f)
            assert good, c0
            owner.cmd("PERMUTE", {"addr": f[3][0], "size": f[3][1]})
            _, c1 = _check(owner, f)
            owner.cmd("PERMUTE", {"addr": f[3][0], "size": f[3][1]})  # swap back
            _, c1b = _check(owner, f)
            owner.cmd("FLIP", {"addr": f[5][0] + 12345})
            _, c2 = _check(owner, f)
            owner.cmd("FLIP", {"addr": f[5][0] + 12345})
            written = sorted(x[0] for x in f)
            gap = None
            for a, b in zip(written, written[1:]):
                g = a + (128 << 10) + 7
                if b - a > (128 << 10) and any(pb <= a and g < pb + ps for pb, ps in ctx.pools):
                    gap = g
                    break
            assert gap is not None
            owner.cmd("POKE", {"addr": gap})
            _, c3 = _check(owner, f)
            ok = (
                c1["bad_bytes"] > 0
                and c1b["bad_bytes"] == 0
                and c2["bad_bytes"] == 1
                and c3["nonzero_outside"] == 1
            )
            return ok, f"swap={c1} swap_back={c1b} flip={c2} gap={c3}"

        run("knob0_identical_routing", t_knob0)
        run("fbw_concurrent_byte_exact", t_fbw)
        run("fbw_equals_nixl", t_same)
        run("gate_routing", t_gate)
        run("export_with_hole", t_hole)
        run("refuse_no_export", t_noexport)
        run("refuse_sleep_rebalance", t_refuse)
        run("fault_first_batch", t_fault("first"))
        run("fault_later_batch", t_fault("later"))
        run("negative_controls", t_neg)
        rc = owner.stop()
        agent.shutdown()
        npass = sum(results.values())
        print(
            f"UT_SUMMARY tests={len(results)} expected={len(EXPECTED_TESTS)} pass={npass} owner_rc={rc}"
        )
        ok = list(results) == EXPECTED_TESTS and npass == len(EXPECTED_TESTS) and rc == 0
        print("ALL_PASS" if ok else "UT_FAIL", flush=True)
        return 0 if ok else 1
    finally:
        # Stop and reap the receiver even when the sender fails before its own stop.
        owner.stop()


def _gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


# Bound on the whole two-process run (the receiver alone may take minutes to build its pools).
_RUN_TIMEOUT_S = 1800


def test_fabric_writer_2gpu():
    if _gpus() < 2:
        pytest.skip("needs 2 GPUs with FABRIC handle support")
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
    if len(sys.argv) > 1 and sys.argv[1] == "owner":
        owner_main()
        sys.exit(0)
    sys.exit(sender_main())
