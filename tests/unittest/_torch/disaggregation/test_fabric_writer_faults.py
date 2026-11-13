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
"""Deterministic fault/interleaving tests of the KV fabric writer (no GPU needed).

- prepare_retire_race: a peer re-registration and a retire sweep land while a request is being
  prepared; the mapping a returned request uses must not be released before that request ends.
- sender_worker_*: a send worker still inside a fabric write when Sender.shutdown's first join expires
  falls back to NIXL after the writer drains; teardown waits for it, or fail-stops if it never ends.
- no_export_reregistration_retires: a mapped peer that registers again without an export goes back to
  NIXL; its old mapping is retired, kept pinned until its in-flight request ends, then released.
- event_record_*: the batch copy is enqueued but recording its completion event fails; the
  request may only fall back to NIXL (or end) after the enqueued copy is drained, and when the
  drain fails it must neither fall back nor unpin its mapping.

The CUDA runtime is replaced by a recorder that tracks which streams still hold copies.
``FBW_MODULE_PATH`` selects the fabric_writer.py under test (default: the installed one).
Run directly: prints UT_RESULT lines and ALL_PASS.
"""

import importlib.util
import os
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

TESTS = [
    "control_success",
    "prepare_retire_race",
    "event_record_drain",
    "event_record_unquiesced",
    "legacy_sender_fail_stop",
    "shutdown_drains_inflight",
    "shutdown_timeout_fail_stop",
    "sender_shutdown_drained_control",
    "sender_shutdown_expired_join",
    "no_export_reregistration_retires",
    "sender_worker_fallback_before_teardown",
    "sender_worker_stuck_after_drain",
]
RUN_BASE, RUN_SIZE, LOCAL_BASE = 1 << 30, 1 << 26, 1 << 40


def _load():
    path = os.environ.get("FBW_MODULE_PATH")
    if not path:
        from tensorrt_llm._torch.disaggregation.native import fabric_writer

        return fabric_writer
    spec = importlib.util.spec_from_file_location("fabric_writer_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Cuda:
    """Recorder standing in for cuda.bindings.runtime."""

    cudaStreamNonBlocking = 1
    cudaEventBlockingSync = 1
    cudaEventDisableTiming = 2
    cudaMemcpySrcAccessOrder = SimpleNamespace(cudaMemcpySrcAccessOrderStream=1)
    cudaMemcpyFlags = SimpleNamespace(cudaMemcpyFlagPreferOverlapWithCompute=1)

    class cudaMemcpyAttributes:  # noqa: N801
        srcAccessOrder = 0
        flags = 0

    def __init__(self, fail_record=False, fail_stream_sync=False):
        self.fail_record = fail_record
        self.fail_stream_sync = fail_stream_sync
        self.pending = set()
        self.event_stream = {}
        self.copies = 0
        self.n = 0

    def cudaSetDevice(self, dev):  # noqa: N802
        return (0,)

    def cudaFree(self, ptr):  # noqa: N802
        return (0,)

    def cudaStreamCreateWithFlags(self, flags):  # noqa: N802
        self.n += 1
        return (0, f"stream{self.n}")

    def cudaEventCreateWithFlags(self, flags):  # noqa: N802
        self.n += 1
        return (0, f"event{self.n}")

    def cudaMemcpyBatchAsync(self, dsts, srcs, sizes, count, attrs, idxs, nattrs, stream):  # noqa: N802
        assert isinstance(sizes, list) and len(dsts) == len(srcs) == len(sizes) == count
        self.copies += 1
        self.pending.add(stream)
        return (0,)

    def cudaEventRecord(self, event, stream):  # noqa: N802
        if self.fail_record:
            return (400,)
        self.event_stream[event] = stream
        return (0,)

    def cudaEventSynchronize(self, event):  # noqa: N802
        self.pending.discard(self.event_stream.get(event))
        return (0,)

    def cudaStreamSynchronize(self, stream):  # noqa: N802
        if self.fail_stream_sync:
            return (700,)
        self.pending.discard(stream)
        return (0,)


class _Agent:
    def __init__(self, cuda):
        self.cuda = cuda
        self.fallbacks = []

    def submit_transfer_requests(self, request):
        self.fallbacks.append({"copy_still_pending": bool(self.cuda.pending)})
        return SimpleNamespace(
            wait=lambda *a: True, is_completed=lambda: True, last_status_str=lambda: "ok"
        )


class _Stopped(Exception):
    pass


def _peer(fw, name="p0"):
    p = fw._PeerMap(name, "digest-old")
    p.rstart = np.asarray([RUN_BASE], dtype=np.uint64)
    p.rend = np.asarray([RUN_BASE + RUN_SIZE], dtype=np.uint64)
    p.lbase = np.asarray([LOCAL_BASE], dtype=np.uint64)
    p.ready = True
    p.released_calls = []
    p.release = lambda: p.released_calls.append(1)
    return p


def _meta(name="p0", n=4):
    return SimpleNamespace(
        sizes=np.full(n, 256 << 10, dtype=np.int64),
        dst_ptrs=RUN_BASE + np.arange(n, dtype=np.int64) * (512 << 10),
        src_ptrs=(1 << 20) + np.arange(n, dtype=np.int64) * (512 << 10),
        dst_device_id=1,
        peer_name=name,
    )


def _writer(fw, agent):
    for k in [k for k in os.environ if k.startswith("TRTLLM_KV_FABRIC_WRITE")]:
        del os.environ[k]
    w = fw.FabricWriter(agent, 0, fw.FabricWriteConfig())
    # Peers are installed by hand; a joinable placeholder keeps the import thread from starting.
    w._importer = SimpleNamespace(join=lambda timeout=None: None)
    return w


def t_control(fw):
    cuda = _Cuda()
    fw.cudart = cuda
    agent = _Agent(cuda)
    w = _writer(fw, agent)
    peer = _peer(fw)
    w._peers["p0"] = peer
    job = w.prepare(_meta(), lambda: "nixl-request")
    assert job is not None
    ok = w.submit(job).wait()
    w.shutdown()
    assert ok and not agent.fallbacks and not cuda.pending and cuda.copies == 1, (
        ok,
        agent.fallbacks,
        cuda.pending,
    )
    assert peer.inflight == 0
    return f"ok=1 fallbacks=0 copies={cuda.copies} pending=0 inflight=0"


def t_race(fw):
    agent = _Agent(_Cuda())
    w = _writer(fw, agent)
    old = _peer(fw)
    w._peers["p0"] = old
    translate = old.translate

    def racing_translate(dst, sizes):
        # The peer re-registers (new export) and a retire sweep runs while this request is
        # being prepared.
        w.add_peer("p0", b"a-different-export")
        w._release_retired()
        return translate(dst, sizes)

    old.translate = racing_translate
    job = w.prepare(_meta(), lambda: "nixl-request")
    released_at_return = len(old.released_calls)
    if job is None:
        # Declining the request is safe as long as nothing stays pinned.
        assert old.inflight == 0
        w._release_retired()
        return f"request declined; old mapping released={len(old.released_calls)}"
    assert job.peer is old
    assert released_at_return == 0, "mapping released while a returned request still uses it"
    with w._lock:
        w._finish_locked(job)
    w._release_retired()
    assert len(old.released_calls) == 1, old.released_calls
    return "returned request kept the mapping pinned; released once after the request ended"


def _event_record_case(fw, fail_stream_sync):
    cuda = _Cuda(fail_record=True, fail_stream_sync=fail_stream_sync)
    fw.cudart = cuda
    agent = _Agent(cuda)
    w = _writer(fw, agent)
    peer = _peer(fw)
    w._peers["p0"] = peer
    job = w.prepare(_meta(), lambda: "nixl-request")
    assert job is not None
    status = w.submit(job)
    job.done.wait(30)
    polled_before_wait = status.is_completed()
    ok = status.wait()
    stops = []

    def hook(reason):
        stops.append(reason)
        raise _Stopped(reason)

    fw._fail_stop_hook = hook
    try:
        w.shutdown()
    except _Stopped:
        pass
    status.polled_before_wait = polled_before_wait
    status.shutdown_stops = stops
    return cuda, agent, peer, job, status, ok


def t_evrec_drain(fw):
    cuda, agent, peer, job, status, ok = _event_record_case(fw, fail_stream_sync=False)
    assert cuda.copies == 1
    assert len(agent.fallbacks) == 1, agent.fallbacks
    assert not agent.fallbacks[0]["copy_still_pending"], "NIXL fallback ran with the copy pending"
    assert ok and peer.inflight == 0, (ok, peer.inflight)
    return "copy drained before the NIXL fallback; fallback ok; mapping unpinned"


def t_evrec_unquiesced(fw):
    cuda, agent, peer, job, status, ok = _event_record_case(fw, fail_stream_sync=True)
    assert not agent.fallbacks, f"fallback admitted with an unproven copy: {agent.fallbacks}"
    assert not ok, "request reported success"
    assert peer.inflight == 1, f"mapping unpinned with a copy possibly in flight ({peer.inflight})"
    assert not status.polled_before_wait, "unproven copy reported complete before wait()"
    assert not status.is_completed(), "unproven copy reported complete after wait()"
    assert not status.wait(), "second wait() reported success"
    assert len(status.shutdown_stops) == 1, "shutdown returned with an unquiesced write"
    return "drain failed: no fallback, request failed, mapping stays pinned, not complete"


def t_legacy_sender(fw):
    """The sender without physical ownership must stop, not report an ordinary failure that lets
    the request's pages be released, when a fabric write ends unquiesced."""
    import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod

    cuda = _Cuda(fail_record=True, fail_stream_sync=True)
    fw.cudart = cuda
    agent = _Agent(cuda)
    w = _writer(fw, agent)
    peer = _peer(fw)
    w._peers["p0"] = peer
    job = w.prepare(_meta(), lambda: "nixl-request")
    assert job is not None
    stops = []

    def hook(reason):
        stops.append((reason, bool(cuda.pending)))
        raise _Stopped(reason)

    assert hasattr(fw, "_fail_stop_hook"), "module has no fail-stop path"
    saved_hook, saved_mod = fw._fail_stop_hook, transfer_mod.fabric_writer
    fw._fail_stop_hook, transfer_mod.fabric_writer = hook, fw
    sender = SimpleNamespace(_fabric=w)
    sender._submit_fabric_job = lambda task, request, on_submitted=None: (
        transfer_mod.Sender._submit_fabric_job(sender, task, request, on_submitted)
    )
    try:
        try:
            outcome = transfer_mod.Sender._submit_transfer(sender, None, 0, job)
        except _Stopped:
            outcome = "stopped"
        try:
            w.shutdown()  # an unquiesced write also stops shutdown
        except _Stopped:
            pass
    finally:
        fw._fail_stop_hook, transfer_mod.fabric_writer = saved_hook, saved_mod
    assert outcome == "stopped", f"legacy sender returned {outcome!r}; the request would retire"
    assert len(stops) == 2 and stops[0][1], stops
    assert not agent.fallbacks and peer.inflight == 1
    return "unquiesced write stops the sender before any result; no fallback; mapping pinned"


class _BlockingCuda(_Cuda):
    """Event waits block until released: a batch stays in flight."""

    def __init__(self):
        super().__init__()
        self.release = __import__("threading").Event()

    def cudaEventSynchronize(self, event):  # noqa: N802
        self.release.wait()
        return super().cudaEventSynchronize(event)


def _inflight_writer(fw):
    cuda = _BlockingCuda()
    fw.cudart = cuda
    agent = _Agent(cuda)
    w = _writer(fw, agent)
    peer = _peer(fw)
    w._peers["p0"] = peer
    job = w.prepare(_meta(), lambda: "nixl-request")
    assert job is not None
    status = w.submit(job)
    import time

    t0 = time.time()
    while not cuda.pending and time.time() - t0 < 10:
        time.sleep(0.01)
    assert cuda.pending, "batch never issued"
    return cuda, agent, w, peer, status


def _run_shutdown(target):
    import threading

    box = {}

    def body():
        try:
            target()
            box["result"] = "returned"
        except _Stopped as e:
            box["result"] = f"stopped: {e}"

    t = threading.Thread(target=body, daemon=True)
    t.start()
    return t, box


def _install_stop_hook(fw):
    stops = []

    def hook(reason):
        stops.append(reason)
        raise _Stopped(reason)

    fw._fail_stop_hook = hook
    return stops


def t_shutdown_drains(fw):
    cuda, agent, w, peer, status = _inflight_writer(fw)
    stops = _install_stop_hook(fw)
    t, box = _run_shutdown(w.shutdown)
    t.join(1.5)
    alive_while_pending = t.is_alive() and bool(cuda.pending)
    cuda.release.set()
    t.join(30)
    assert alive_while_pending, "shutdown returned while a batch was in flight"
    assert box.get("result") == "returned" and not stops, (box, stops)
    assert not cuda.pending and status.wait()
    return "shutdown blocked until the in-flight batch landed, then returned"


def t_shutdown_timeout(fw):
    cuda, agent, w, peer, status = _inflight_writer(fw)
    stops = _install_stop_hook(fw)
    w._shutdown_timeout_s = 1.0
    t, box = _run_shutdown(w.shutdown)
    t.join(20)
    pending_at_stop = bool(cuda.pending)
    cuda.release.set()
    assert box.get("result", "").startswith("stopped"), f"shutdown did not fail-stop: {box}"
    assert pending_at_stop and len(stops) == 1, (pending_at_stop, stops)
    return "copies still in flight past the deadline: fail-stop, never a normal return"


def _sender_shell(w):
    import threading

    import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod

    calls = []
    shell = SimpleNamespace(
        _retirement_watchdog=None,
        _sessions_lock=threading.Lock(),
        _shutdown=False,
        _shutdown_requested=False,
        _enforce_physical_ownership=False,
        _sessions={},
        _pending_settlements=[],
        _messenger=SimpleNamespace(stop=lambda: calls.append("messenger_stop")),
        _send_task_queues=[],
        _worker_threads=[],
        _fabric=w,
        _loaded_remote_agents_lock=threading.Lock(),
        _loaded_remote_agents=set(),
        _agent=SimpleNamespace(invalidate_remote_agent=lambda name: None),
        _dealers={},
    )
    return transfer_mod, shell


def t_sender_drained(fw):
    cuda, agent, w, peer, status = _inflight_writer(fw)
    stops = _install_stop_hook(fw)
    transfer_mod, shell = _sender_shell(w)
    t, box = _run_shutdown(lambda: transfer_mod.Sender.shutdown(shell))
    t.join(1.5)
    alive_while_pending = t.is_alive() and bool(cuda.pending)
    cuda.release.set()
    t.join(30)
    assert alive_while_pending, "Sender.shutdown returned while a batch was in flight"
    assert box.get("result") == "returned" and shell._shutdown and not stops, (box, stops)
    assert not cuda.pending
    return "Sender.shutdown (the caller of pool deregistration) returns only after the drain"


def t_sender_expired(fw):
    cuda, agent, w, peer, status = _inflight_writer(fw)
    stops = _install_stop_hook(fw)
    w._shutdown_timeout_s = 1.0
    transfer_mod, shell = _sender_shell(w)
    t, box = _run_shutdown(lambda: transfer_mod.Sender.shutdown(shell))
    t.join(20)
    pending_at_stop = bool(cuda.pending)
    cuda.release.set()
    assert box.get("result", "").startswith("stopped"), f"Sender.shutdown returned: {box}"
    assert pending_at_stop and not shell._shutdown and len(stops) == 1, (
        pending_at_stop,
        shell._shutdown,
        stops,
    )
    return "expired drain through Sender.shutdown: fail-stop before teardown can continue"


def t_no_export_retires(fw):
    cuda = _Cuda()
    fw.cudart = cuda
    agent = _Agent(cuda)
    w = _writer(fw, agent)
    old = _peer(fw)
    w._peers["p0"] = old
    assert w.peer_ready("p0")
    inflight = w.prepare(_meta(), lambda: "nixl-request")
    assert inflight is not None and old.inflight == 1

    w.add_peer("p0", None)

    assert not w.peer_ready("p0") and old.retired
    # Later requests take NIXL, never the old mapping.
    assert w.prepare(_meta(), lambda: "nixl-request") is None
    # Still pinned by the request prepared before the re-registration.
    assert old.released_calls == []
    assert w.submit(inflight).wait()
    w._release_retired()
    assert old.inflight == 0 and old.released_calls == [1]
    w.shutdown()
    return "re-registration without export: peer -> NIXL, old mapping released after its request"


def _sender_with_late_worker(fw, finishes):
    """A Sender shell whose one worker is still busy after the first join, and the event log."""
    import threading

    cuda = _Cuda()
    fw.cudart = cuda
    w = _writer(fw, _Agent(cuda))
    transfer_mod, shell = _sender_shell(w)
    calls, drained, release = [], threading.Event(), threading.Event()
    writer_shutdown = w.shutdown

    def shutdown_and_mark():
        writer_shutdown()
        drained.set()

    w.shutdown = shutdown_and_mark

    def worker():
        # The NIXL fallback of a fabric write that failed once the writer drained.
        drained.wait(30)
        if finishes:
            calls.append("worker_fallback_done")
        else:
            release.wait(60)

    thread = threading.Thread(target=worker, name="send-worker-0", daemon=True)
    thread.start()
    shell._worker_threads = [thread]
    shell._loaded_remote_agents = {"peer0"}
    shell._agent = SimpleNamespace(invalidate_remote_agent=lambda name: calls.append("invalidate"))
    return transfer_mod, shell, calls, release


def _with_short_sender_joins(transfer_mod, fw, body):
    saved = (
        transfer_mod._SENDER_WORKER_JOIN_S,
        transfer_mod._SENDER_FABRIC_WORKER_DRAIN_S,
        transfer_mod.fabric_writer,
    )
    transfer_mod._SENDER_WORKER_JOIN_S = 0.05
    transfer_mod._SENDER_FABRIC_WORKER_DRAIN_S = 1.0
    transfer_mod.fabric_writer = fw
    try:
        return body()
    finally:
        (
            transfer_mod._SENDER_WORKER_JOIN_S,
            transfer_mod._SENDER_FABRIC_WORKER_DRAIN_S,
            transfer_mod.fabric_writer,
        ) = saved


def t_sender_worker_fallback(fw):
    transfer_mod, shell, calls, _ = _sender_with_late_worker(fw, finishes=True)
    stops = _install_stop_hook(fw)

    def body():
        transfer_mod.Sender.shutdown(shell)

    _with_short_sender_joins(transfer_mod, fw, body)
    assert calls == ["worker_fallback_done", "invalidate"] and shell._shutdown, calls
    assert not stops, stops
    return "worker's NIXL fallback finished before the remote agents were invalidated"


def t_sender_worker_stuck(fw):
    transfer_mod, shell, calls, release = _sender_with_late_worker(fw, finishes=False)
    stops = _install_stop_hook(fw)

    def body():
        t, box = _run_shutdown(lambda: transfer_mod.Sender.shutdown(shell))
        t.join(20)
        return box

    try:
        box = _with_short_sender_joins(transfer_mod, fw, body)
    finally:
        release.set()
    assert box.get("result", "").startswith("stopped"), box
    assert len(stops) == 1 and "invalidate" not in calls and not shell._shutdown, (stops, calls)
    return "a worker still running after the drain fail-stops before any teardown"


# Module state the cases replace (the CUDA runtime stand-in, the fail-stop hook) and the environment
# _writer clears; main restores all of it, so later tests in the process see the real module.
_PATCHED_ATTRS = ("cudart", "_fail_stop_hook")
_ENV_PREFIX = "TRTLLM_KV_FABRIC_WRITE"


def main():
    fw = _load()
    saved_attrs = {name: getattr(fw, name) for name in _PATCHED_ATTRS if hasattr(fw, name)}
    saved_env = {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIX)}
    try:
        return _run_cases(fw)
    finally:
        for name, value in saved_attrs.items():
            setattr(fw, name, value)
        for k in [k for k in os.environ if k.startswith(_ENV_PREFIX)]:
            del os.environ[k]
        os.environ.update(saved_env)


def _run_cases(fw):
    results = {}
    fns = [
        t_control,
        t_race,
        t_evrec_drain,
        t_evrec_unquiesced,
        t_legacy_sender,
        t_shutdown_drains,
        t_shutdown_timeout,
        t_sender_drained,
        t_sender_expired,
        t_no_export_retires,
        t_sender_worker_fallback,
        t_sender_worker_stuck,
    ]
    for name, fn in zip(TESTS, fns):
        try:
            detail, ok = fn(fw), True
        except Exception:
            detail, ok = traceback.format_exc().replace("\n", " | "), False
        results[name] = ok
        print(f"UT_RESULT {name} {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    npass = sum(results.values())
    print(f"UT_SUMMARY tests={len(results)} pass={npass} module={fw.__file__}", flush=True)
    good = list(results) == TESTS and npass == len(TESTS)
    print("ALL_PASS" if good else "UT_FAIL", flush=True)
    return 0 if good else 1


def test_fabric_writer_faults():
    fw = _load()
    before = {name: getattr(fw, name) for name in _PATCHED_ATTRS if hasattr(fw, name)}
    env_before = {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIX)}
    assert main() == 0
    # Nothing leaks into later tests in this process.
    assert {name: getattr(fw, name) for name in before} == before
    assert {k: v for k, v in os.environ.items() if k.startswith(_ENV_PREFIX)} == env_before


if __name__ == "__main__":
    sys.exit(main())
