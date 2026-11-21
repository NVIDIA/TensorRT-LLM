# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""drain_captured_hostfuncs: host functions of a replayed graph vs. GIL-holding waits.

A captured host function takes the GIL when the device reaches it. If the executor then
enters a native call that holds the GIL while it waits on the stream, the process
deadlocks. The GPU test reproduces that in a child process (a GIL-holding
cuStreamSynchronize through ctypes.PyDLL stands in for such a call) and checks that
draining first avoids it.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time

import pytest
import torch

from tensorrt_llm._torch import hostfunc

DEADLINE_AFTER_WAIT_S = 30


class _Event:

    def __init__(self):
        self.synchronized = 0

    def record(self, *args):
        pass

    def synchronize(self):
        self.synchronized += 1


@pytest.fixture
def stub_events(monkeypatch):
    events = []

    def make_event():
        events.append(_Event())
        return events[-1]

    monkeypatch.setattr(hostfunc, "_replay_event", None)
    monkeypatch.setattr(torch.cuda, "Event", make_event)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return events


def test_drain_without_replay_is_a_no_op(stub_events):
    assert not hostfunc.hostfuncs_pending()
    hostfunc.drain_captured_hostfuncs()
    assert stub_events == []


def test_drain_waits_once_for_the_last_replay(stub_events):
    hostfunc.record_hostfunc_graph_replay()
    hostfunc.record_hostfunc_graph_replay()
    first, last = stub_events
    hostfunc.drain_captured_hostfuncs()
    assert (first.synchronized, last.synchronized) == (0, 1)
    assert not hostfunc.hostfuncs_pending()
    hostfunc.drain_captured_hostfuncs()
    assert last.synchronized == 1


def test_drain_is_skipped_while_capturing(stub_events, monkeypatch):
    hostfunc.record_hostfunc_graph_replay()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    hostfunc.drain_captured_hostfuncs()
    assert stub_events[0].synchronized == 0
    assert hostfunc.hostfuncs_pending()


def _child(drain: bool) -> None:
    libcuda = ctypes.PyDLL("libcuda.so.1")  # PyDLL keeps the GIL during the call.
    runs = []
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.cuda._sleep(2_000_000_000)
        hostfunc.launch_hostfunc(lambda: runs.append(1))
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        graph.replay()
        hostfunc.record_hostfunc_graph_replay()
        if drain:
            hostfunc.drain_captured_hostfuncs()
        print("WAITING", flush=True)
        rc = libcuda.cuStreamSynchronize(ctypes.c_void_p(stream.cuda_stream))
    print(f"DONE rc={rc} runs={len(runs)}", flush=True)


def _run_child(drain: bool) -> str:
    proc = subprocess.Popen([sys.executable, __file__, "--child", str(int(drain))],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True)
    lines = []
    threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True).start()
    waiting_since = None
    while proc.poll() is None:
        if waiting_since is None and "WAITING\n" in lines:
            waiting_since = time.monotonic()
        if waiting_since is not None and time.monotonic() - waiting_since > DEADLINE_AFTER_WAIT_S:
            proc.kill()
            proc.wait()
            return "deadlocked"
        time.sleep(0.2)
    time.sleep(0.5)
    return "completed" if "DONE rc=0 runs=1\n" in lines else "failed"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("drain, expected", [(True, "completed"), (False, "deadlocked")])
def test_gil_holding_wait_after_replay(drain, expected):
    assert _run_child(drain) == expected


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--child":
    _child(bool(int(sys.argv[2])))
