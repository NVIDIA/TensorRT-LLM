import threading

import torch

from tensorrt_llm._torch.hostfunc import HOSTFUNC_USER_DATA_HANDLES, hostfunc


def test_hostfunc():

    @hostfunc
    def increase(x: torch.Tensor):
        x.add_(1)

    x = torch.zeros(10, dtype=torch.int32)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(5):
            increase(x)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        increase(x)
        increase(x)
    torch.cuda.synchronize()

    with torch.cuda.stream(stream):
        for _ in range(10):
            g.replay()
    torch.cuda.synchronize()

    assert (x == 25).all().item()
    assert len(HOSTFUNC_USER_DATA_HANDLES) == 2


def test_hostfunc_gil_release_no_deadlock():
    """Regression guard for the guided-decoding x host-func GIL deadlock.

    launch_hostfunc must not hold the GIL across cudaLaunchHostFunc. The driver
    serializes host-function dispatch on a stream, so a launch issued while an
    earlier callback is in flight can busy-wait in the driver. That in-flight
    callback runs through cudaHostFuncTrampoline, which acquires the GIL to call
    Python (e.g. xgrammar in guided decoding); if the launching thread holds the
    GIL while it busy-waits, the callback can never acquire it and the two
    deadlock. See cpp/tensorrt_llm/nanobind/runtime/hostfunc.cpp::launchHostFunc.

    Best-effort guard: the fixed build completes promptly; a build that holds the
    GIL across the enqueue hangs and is caught by the join timeout.
    """
    started = threading.Event()
    gate = threading.Event()

    @hostfunc
    def blocker():
        # Runs on a CUDA driver worker with the GIL held by the trampoline.
        # gate.wait() drops the GIL while waiting and must reacquire it to
        # return -- mimicking xgrammar reacquiring the GIL after native work.
        started.set()
        gate.wait(20)

    @hostfunc
    def noop():
        pass

    stream = torch.cuda.Stream()

    def issue():
        with torch.cuda.stream(stream):
            blocker()
            noop()  # second same-stream launch while blocker is in flight
        torch.cuda.synchronize()

    worker = threading.Thread(target=issue, name="hostfunc-issue")
    worker.start()
    # Let the blocker callback begin (fixed build: the launching thread released
    # the GIL so the callback can run), then release it so it can drain.
    started.wait(20)
    gate.set()

    worker.join(60)
    assert not worker.is_alive(), (
        "launch_hostfunc deadlocked: the GIL was held across cudaLaunchHostFunc")
