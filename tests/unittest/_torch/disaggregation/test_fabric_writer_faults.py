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
"""Deterministic fault and interleaving tests of the KV fabric writer (no GPU needed).

The CUDA runtime is replaced by a recorder that tracks which streams still hold copies.
"""

import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native import fabric_writer

pytestmark = pytest.mark.cpu_only

RUN_BASE, RUN_SIZE, LOCAL_BASE = 1 << 30, 1 << 26, 1 << 40


@pytest.fixture
def fw(monkeypatch):
    """The fabric_writer module, with its patched state and environment restored afterwards."""
    for name in ("cudart", "_fail_stop_hook"):
        monkeypatch.setattr(fabric_writer, name, getattr(fabric_writer, name, None), raising=False)
    for k in [k for k in os.environ if k.startswith("TRTLLM_KV_FABRIC_WRITE")]:
        monkeypatch.delenv(k)
    return fabric_writer


class _Cuda:
    """Recorder standing in for cuda.bindings.runtime."""

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

    def cudaSetDevice(self, *_):  # noqa: N802
        return (0,)

    cudaFree = cudaSetDevice  # noqa: N815

    def cudaStreamCreateWithFlags(self, flags):  # noqa: N802
        self.n += 1
        return (0, f"handle{self.n}")

    cudaEventCreateWithFlags = cudaStreamCreateWithFlags  # noqa: N815

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


class _BlockingCuda(_Cuda):
    """Event waits block until released: a batch stays in flight."""

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def cudaEventSynchronize(self, event):  # noqa: N802
        self.release.wait()
        return super().cudaEventSynchronize(event)


class _Agent:
    """The NIXL agent: records each fallback and whether a copy was still pending then."""

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


def _meta():
    offsets = np.arange(4, dtype=np.int64) * (512 << 10)
    return SimpleNamespace(
        sizes=np.full(4, 256 << 10, dtype=np.int64),
        dst_ptrs=RUN_BASE + offsets,
        src_ptrs=(1 << 20) + offsets,
        dst_device_id=1,
        peer_name="p0",
    )


def _writer(fw, cuda, prepare=True):
    """A writer with one mapped peer "p0" (and one prepared request); returns (w, agent, peer, job)."""
    fw.cudart = cuda
    agent = _Agent(cuda)
    w = fw.FabricWriter(agent, 0, fw.FabricWriteConfig())
    # Peers are installed by hand; a joinable placeholder keeps the import thread from starting.
    w._importer = SimpleNamespace(join=lambda timeout=None: None)
    peer = fw._PeerMap("p0", "digest-old")
    peer.rstart, peer.rend, peer.lbase = (
        np.asarray([x], dtype=np.uint64) for x in (RUN_BASE, RUN_BASE + RUN_SIZE, LOCAL_BASE)
    )
    peer.ready = True
    peer.released_calls = []
    peer.release = lambda: peer.released_calls.append(1)
    w._peers["p0"] = peer
    job = w.prepare(_meta(), lambda: "nixl-request") if prepare else None
    assert job is not None or not prepare
    return w, agent, peer, job


def _install_stop_hook(fw, cuda=None):
    stops = []

    def hook(reason):
        stops.append(reason if cuda is None else (reason, bool(cuda.pending)))
        raise _Stopped(reason)

    fw._fail_stop_hook = hook
    return stops


def _run_shutdown(target):
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


def _sender_shell(w):
    return SimpleNamespace(
        _retirement_watchdog=None,
        _sessions_lock=threading.Lock(),
        _shutdown=False,
        _shutdown_requested=False,
        _enforce_physical_ownership=False,
        _sessions={},
        _pending_settlements=[],
        _messenger=SimpleNamespace(stop=lambda: None),
        _send_task_queues=[],
        _worker_threads=[],
        _fabric=w,
        _loaded_remote_agents_lock=threading.Lock(),
        _loaded_remote_agents=set(),
        _agent=SimpleNamespace(invalidate_remote_agent=lambda name: None),
        _dealers={},
    )


def test_prepare_retire_race_keeps_the_mapping_pinned(fw):
    """A re-registration and a retire sweep during prepare never release a mapping in use."""
    w, _, old, _ = _writer(fw, _Cuda(), prepare=False)
    translate = old.translate

    def racing_translate(dst, sizes):
        w.add_peer("p0", b"a-different-export")
        w._release_retired()
        return translate(dst, sizes)

    old.translate = racing_translate
    job = w.prepare(_meta(), lambda: "nixl-request")
    released_at_return = len(old.released_calls)
    if job is None:
        # Declining the request is safe as long as nothing stays pinned.
        assert old.inflight == 0
        return
    assert job.peer is old
    assert released_at_return == 0, "mapping released while a returned request still uses it"
    with w._lock:
        w._finish_locked(job)
    w._release_retired()
    assert len(old.released_calls) == 1, old.released_calls


@pytest.mark.parametrize("drain_fails", [False, True], ids=["drained", "unquiesced"])
def test_event_record_failure(fw, drain_fails):
    """A failed event record: NIXL fallback only after the drain; a failed drain fail-stops."""
    cuda = _Cuda(fail_record=True, fail_stream_sync=drain_fails)
    w, agent, peer, job = _writer(fw, cuda)
    status = w.submit(job)
    job.done.wait(30)
    polled_before_wait = status.is_completed()
    ok = status.wait()
    stops = _install_stop_hook(fw)
    try:
        w.shutdown()
    except _Stopped:
        pass
    assert cuda.copies == 1
    if not drain_fails:
        assert len(agent.fallbacks) == 1, agent.fallbacks
        assert not agent.fallbacks[0]["copy_still_pending"], "fallback ran with the copy pending"
        assert ok and peer.inflight == 0, (ok, peer.inflight)
        return
    assert not agent.fallbacks, f"fallback admitted with an unproven copy: {agent.fallbacks}"
    assert not ok, "request reported success"
    assert peer.inflight == 1, "mapping unpinned with a copy possibly in flight"
    assert not polled_before_wait and not status.is_completed() and not status.wait()
    assert len(stops) == 1, "shutdown returned with an unquiesced write"


def test_unquiesced_write_stops_the_sender_without_physical_ownership(fw):
    """The sender stops instead of reporting a failure that would let the pages be released."""
    cuda = _Cuda(fail_record=True, fail_stream_sync=True)
    w, agent, peer, job = _writer(fw, cuda)
    stops = _install_stop_hook(fw, cuda)
    sender = SimpleNamespace(_fabric=w)
    sender._submit_fabric_job = lambda task, request, on_submitted=None: (
        transfer_mod.Sender._submit_fabric_job(sender, task, request, on_submitted)
    )
    with pytest.raises(_Stopped):
        transfer_mod.Sender._submit_transfer(sender, None, 0, job)
    with pytest.raises(_Stopped):
        w.shutdown()  # an unquiesced write also stops shutdown
    assert len(stops) == 2 and stops[0][1], stops
    assert not agent.fallbacks and peer.inflight == 1


def test_rejected_retry_of_a_quiesced_write_fails_without_a_submission(fw):
    """A retry whose remote descriptors are gone is rejected before the agent sees it."""
    cuda = _Cuda(fail_record=True)  # the copy fails, and the drain quiesces it
    w, agent, peer, _ = _writer(fw, cuda, prepare=False)
    stops = _install_stop_hook(fw)
    sender = SimpleNamespace(
        _fabric=w, _agent=SimpleNamespace(check_remote_descs=lambda name, descs: False)
    )
    sender._submit_fabric_job = lambda task, request, on_submitted=None: (
        transfer_mod.Sender._submit_fabric_job(sender, task, request, on_submitted)
    )
    write = SimpleNamespace(op=transfer_mod.TransferOp.WRITE, remote_name="p0", dst_descs="d")

    def retry():
        transfer_mod.Sender._check_write_descs(sender, (write,))
        return write

    job = w.prepare(_meta(), retry)
    with pytest.raises(transfer_mod._TransferNotSubmittedError):
        transfer_mod.Sender._submit_transfer(sender, None, 0, job)
    w.shutdown()
    assert not agent.fallbacks and not stops
    assert peer.inflight == 0


@pytest.mark.parametrize("expired", [False, True], ids=["drains", "expired"])
def test_sender_shutdown_with_a_batch_in_flight(fw, expired):
    """Sender.shutdown returns only after the drain, or fail-stops once the drain deadline passes."""
    cuda = _BlockingCuda()
    w, _, _, job = _writer(fw, cuda)
    w.submit(job)
    t0 = time.time()
    while not cuda.pending and time.time() - t0 < 10:
        time.sleep(0.01)
    assert cuda.pending, "batch never issued"
    stops = _install_stop_hook(fw)
    if expired:
        w._shutdown_timeout_s = 1.0
    shell = _sender_shell(w)
    t, box = _run_shutdown(lambda: transfer_mod.Sender.shutdown(shell))
    if expired:
        t.join(20)
        pending_at_stop = bool(cuda.pending)
        cuda.release.set()
        # A real fail-stop ends the process; here the released batch lets the dispatcher exit.
        w._dispatcher.join(30)
        assert not w._dispatcher.is_alive(), "dispatcher still running after its batch landed"
        assert box.get("result", "").startswith("stopped"), f"Sender.shutdown returned: {box}"
        assert pending_at_stop and not shell._shutdown and len(stops) == 1
        return
    t.join(1.5)
    alive_while_pending = t.is_alive() and bool(cuda.pending)
    cuda.release.set()
    t.join(30)
    assert alive_while_pending, "Sender.shutdown returned while a batch was in flight"
    assert box.get("result") == "returned" and shell._shutdown and not stops, (box, stops)
    assert not cuda.pending


def test_reregistration_without_export_retires_the_mapping(fw):
    cuda = _Cuda()
    w, agent, old, inflight = _writer(fw, cuda)
    assert w.peer_ready("p0") and old.inflight == 1

    w.add_peer("p0", None)

    assert not w.peer_ready("p0") and old.retired
    # Later requests take NIXL, never the old mapping.
    assert w.prepare(_meta(), lambda: "nixl-request") is None
    # Still pinned by the request prepared before the re-registration.
    assert old.released_calls == []
    # The in-flight request still completes as a fabric copy.
    assert w.submit(inflight).wait()
    assert not agent.fallbacks and not cuda.pending and cuda.copies == 1
    w._release_retired()
    assert old.inflight == 0 and old.released_calls == [1]
    w.shutdown()


@pytest.mark.parametrize("pinned", [False, True], ids=["idle", "job_in_flight"])
def test_remove_peer_releases_the_mapping_once_idle(fw, pinned):
    """A peer that left (fresh instance names never re-register) does not stay mapped."""
    cuda = _Cuda()
    w, _, peer, job = _writer(fw, cuda, prepare=pinned)
    w.remove_peer("p0")
    assert not w.peer_ready("p0") and peer.retired
    if pinned:
        assert peer.released_calls == []
        assert w.submit(job).wait()
        w._release_retired()
    assert peer.released_calls == [1]
    w.remove_peer("unknown")  # no-op
    w.shutdown()


def test_sender_unregister_message_removes_the_fabric_peer(fw):
    removed = []
    shell = SimpleNamespace(
        _loaded_remote_agents_lock=threading.Lock(),
        _fabric=SimpleNamespace(remove_peer=removed.append),
    )
    transfer_mod.Sender._unregister_peer_rank(
        shell, [transfer_mod.MessageType.UNREGISTER_RANK_INFO, b"gen-instance", b"3"]
    )
    assert removed == ["gen-instance3"]


@pytest.mark.parametrize(
    "case", ["fallback_done", "stuck_in_fabric_job", "stuck_in_plain_transfer"]
)
def test_sender_worker_still_busy_after_the_first_join(fw, monkeypatch, case):
    """A worker inside a fabric job (its NIXL fallback included) must end before teardown, and one
    that does not fail-stops: no remote agent is invalidated under its transfer. A worker blocked in
    an ordinary transfer-agent write is logged and left behind, as without the writer."""
    monkeypatch.setattr(transfer_mod, "_SENDER_WORKER_JOIN_S", 0.05)
    monkeypatch.setattr(transfer_mod, "_SENDER_FABRIC_WORKER_DRAIN_S", 1.0)
    w, _, _, _ = _writer(fw, _Cuda(), prepare=False)
    shell = _sender_shell(w)
    shell._fabric_job_threads = set()
    calls, drained, release, started = [], threading.Event(), threading.Event(), threading.Event()
    writer_shutdown = w.shutdown

    def shutdown_and_mark():
        writer_shutdown()
        drained.set()

    w.shutdown = shutdown_and_mark

    def worker():
        in_fabric = case != "stuck_in_plain_transfer"
        if in_fabric:
            shell._fabric_job_threads.add(threading.get_ident())
        started.set()
        try:
            if case == "fallback_done":
                # The NIXL fallback of a fabric write that failed once the writer drained.
                drained.wait(30)
                calls.append("worker_fallback_done")
            else:
                release.wait(60)
        finally:
            shell._fabric_job_threads.discard(threading.get_ident())

    thread = threading.Thread(target=worker, name="send-worker-0", daemon=True)
    thread.start()
    started.wait(10)
    shell._worker_threads = [thread]
    shell._loaded_remote_agents = {"peer0"}
    shell._agent = SimpleNamespace(invalidate_remote_agent=lambda name: calls.append("invalidate"))
    stops = _install_stop_hook(fw)
    try:
        t, box = _run_shutdown(lambda: transfer_mod.Sender.shutdown(shell))
        t.join(20)
    finally:
        release.set()
        thread.join(5)
    if case == "fallback_done":
        assert box.get("result") == "returned" and not stops, (box, stops)
        assert calls == ["worker_fallback_done", "invalidate"] and shell._shutdown, calls
    elif case == "stuck_in_fabric_job":
        assert box.get("result", "").startswith("stopped"), box
        assert len(stops) == 1 and "invalidate" not in calls and not shell._shutdown
    else:
        assert box.get("result") == "returned" and not stops, (box, stops)
        assert calls == ["invalidate"] and shell._shutdown


def test_a_stopped_writer_builds_no_job(fw):
    """Writes queued behind the shutdown take the transfer agent directly, never a fabric job."""
    w, _, _, _ = _writer(fw, _Cuda(), prepare=False)
    w.shutdown()
    assert w.prepare(_meta(), lambda: "nixl-request") is None
    assert w.stats().get("inelig_stopped") == 1


@pytest.mark.parametrize(
    "model_default, raw, expected",
    [(False, None, False), (True, None, True), (True, "0", False), (False, "1", True)],
    ids=["off", "model_opt_in", "variable_turns_off", "variable_turns_on"],
)
def test_fabric_write_follows_the_model_unless_the_variable_is_set(
    fw, monkeypatch, model_default, raw, expected
):
    monkeypatch.setattr(fw, "_model_default", False)
    fw.set_model_default(model_default)
    if raw is not None:
        monkeypatch.setenv("TRTLLM_KV_FABRIC_WRITE", raw)
    assert fw.fabric_write_enabled() is expected


@pytest.mark.parametrize(
    "raw, expected",
    [("off", False), ("FALSE", False), ("no", False), (" On ", True), ("yes", True)],
)
def test_fabric_write_switch_spellings(fw, monkeypatch, raw, expected):
    monkeypatch.setattr(fw, "_model_default", not expected)
    monkeypatch.setenv("TRTLLM_KV_FABRIC_WRITE", raw)
    assert fw.fabric_write_enabled() is expected


def test_an_unknown_fabric_write_value_fails(fw, monkeypatch):
    monkeypatch.setenv("TRTLLM_KV_FABRIC_WRITE", "disable")
    with pytest.raises(ValueError, match="TRTLLM_KV_FABRIC_WRITE"):
        fw.fabric_write_enabled()


def test_an_idle_peer_is_unmapped_and_reimported_on_its_next_write(fw, monkeypatch):
    """A peer that died without unregistering stops pinning its pool once idle; a live peer's next
    write re-imports the mapping and takes the transfer agent meanwhile."""
    monkeypatch.setenv("TRTLLM_KV_FABRIC_WRITE_IDLE_UNMAP_S", "5")
    w, _, peer, job = _writer(fw, _Cuda())
    peer.blob = b"export"
    assert w.submit(job).wait()
    w._release_retired()
    clock = [time.monotonic()]
    monkeypatch.setattr(fw, "_peer_clock", lambda: clock[0])
    peer.last_used = clock[0]
    w._last_idle_sweep = clock[0]

    # Not idle long enough: the mapping stays.
    clock[0] += 4.0
    assert w.prepare(_meta(), lambda: "nixl-request") is not None
    w._unpin(peer)
    assert peer.released_calls == [] and w.peer_ready("p0")

    # Idle past the bound: the next write's sweep unmaps it and queues a fresh import.
    clock[0] += 6.0
    assert w.prepare(_meta(), lambda: "nixl-request") is None
    assert peer.retired and peer.released_calls == [1]
    remapped = w._peers["p0"]
    assert remapped is not peer and not remapped.ready and remapped.blob == b"export"
    assert list(w._import_q) == [(remapped, b"export")]
    assert w.stats()["peers_idle_unmapped"] == 1
    w.shutdown()


def test_an_idle_peer_that_never_writes_again_stays_unmapped(fw, monkeypatch):
    monkeypatch.setenv("TRTLLM_KV_FABRIC_WRITE_IDLE_UNMAP_S", "5")
    w, _, peer, _ = _writer(fw, _Cuda(), prepare=False)
    peer.blob = b"export"
    clock = [time.monotonic() + 60.0]
    monkeypatch.setattr(fw, "_peer_clock", lambda: clock[0])
    other = _meta()
    other.peer_name = "p1"
    assert w.prepare(other, lambda: "nixl-request") is None
    assert peer.released_calls == [1] and "p0" not in w._peers and w._dormant == {"p0": b"export"}
    # A peer that leaves while dormant is forgotten.
    w.remove_peer("p0")
    assert w._dormant == {}
    w.shutdown()


def test_idle_unmap_off_keeps_the_mapping(fw, monkeypatch):
    monkeypatch.setenv("TRTLLM_KV_FABRIC_WRITE_IDLE_UNMAP_S", "0")
    w, _, peer, _ = _writer(fw, _Cuda(), prepare=False)
    peer.blob = b"export"
    clock = [time.monotonic() + 1e6]
    monkeypatch.setattr(fw, "_peer_clock", lambda: clock[0])
    job = w.prepare(_meta(), lambda: "nixl-request")
    assert job is not None and peer.released_calls == []
    assert w.submit(job).wait()
    w.shutdown()


@pytest.mark.parametrize("fenced", [False, True], ids=["agent_ready", "recovery_pending"])
def test_a_fabric_fallback_takes_the_senders_recovery_path(fw, fenced):
    """A failed fabric write falls back through the sender's own submission path.

    With agent recovery on, that path checks the agent generation and fencing; the writer's agent, which recovery
    may have replaced, never sees the fallback.
    """
    import types

    cuda = _Cuda(fail_record=True)
    w, agent, _, job = _writer(fw, cuda)
    calls = []

    def direct(task, peer_rank, requests, on_submitted, agent_generation, session=None):
        calls.append((requests, agent_generation))
        if fenced:
            raise transfer_mod._TransferNotSubmittedError("NIXL agent recovery is pending")
        return True, None

    shell = SimpleNamespace(
        _fabric=w,
        _recover_agent=lambda generation: None,
        _owns_task=lambda task: False,
        _submit_direct_transfer=direct,
    )
    for name in ("_submit_transfer", "_submit_fabric_job"):
        setattr(shell, name, types.MethodType(getattr(transfer_mod.Sender, name), shell))
    task = SimpleNamespace(_unique_rid=7)
    try:
        if fenced:
            with pytest.raises(transfer_mod._TransferNotSubmittedError):
                shell._submit_transfer(task, 0, job, None, agent_generation=5)
        else:
            assert shell._submit_transfer(task, 0, job, None, agent_generation=5) == (True, None)
    finally:
        w.shutdown()
    assert calls == [(("nixl-request",), 5)]
    assert agent.fallbacks == []  # the writer's own agent never saw the fallback
