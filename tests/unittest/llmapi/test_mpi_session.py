# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess  # nosec B404
import sys
import threading
from subprocess import PIPE, Popen
from typing import Literal

cur_dir = os.path.dirname(os.path.abspath(__file__))

import pytest

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.mpi_session import (_DEFAULT_IDENTITY_TIMEOUT,
                                             MPINodeState, MpiPoolSession,
                                             RemoteMpiCommSessionClient,
                                             _identity_bootstrap_timeout,
                                             split_mpi_env)

# isort: off
sys.path.append(os.path.join(cur_dir, '..'))
# isort: on


def task0():
    if MPINodeState.state is None:
        MPINodeState.state = 0
    MPINodeState.state += 1
    return MPINodeState.state


@pytest.fixture(autouse=True)
def _enable_mpi(monkeypatch):
    monkeypatch.delenv("TLLM_DISABLE_MPI", raising=False)


@pytest.mark.cpu_only
@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
def test_mpi_session_basic():
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    n_workers = 4
    executor = MpiPoolSession(n_workers)
    results = executor.submit_sync(task0)
    assert results == [1, 1, 1, 1], results

    results = executor.submit_sync(task0)
    assert results == [2, 2, 2, 2], results


def simple_task(x):
    print(f"** simple_task {x} returns {x * 2}\n", "green")
    res = x * 2
    print(f"simple_task {x} returns {res}")


def run_client(server_addr, values_to_process, hmac_key: bytes):
    """Function to run in a separate process that creates a client and submits tasks"""
    try:
        client = RemoteMpiCommSessionClient(server_addr, hmac_key=hmac_key)

        for val in values_to_process:
            print(f"Client Submitting task for value {val}")
            client.submit(simple_task, val)

        client.shutdown()

    except Exception as e:
        return f"Error in client: {str(e)}"


@pytest.mark.cpu_only
@pytest.mark.parametrize("task_type", ["submit", "submit_sync"])
def test_remote_mpi_session(task_type: Literal["submit", "submit_sync"]):
    """Test RemoteMpiPoolSessionClient and RemoteMpiPoolSessionServer interaction"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(cur_dir, "_test_remote_mpi_session.sh")
    assert os.path.exists(test_file), f"Test file {test_file} does not exist"
    command = ["bash", test_file, task_type]
    print(' '.join(command))

    with Popen(command,
               env=os.environ,
               stdout=PIPE,
               stderr=PIPE,
               bufsize=1,
               start_new_session=True,
               universal_newlines=True,
               cwd=os.path.dirname(os.path.abspath(__file__))) as process:

        # Function to read from a stream and write to output
        def read_stream(stream, output_stream):
            for line in stream:
                output_stream.write(line)
                output_stream.flush()

        # Create threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=read_stream,
                                         args=(process.stdout, sys.stdout))
        stderr_thread = threading.Thread(target=read_stream,
                                         args=(process.stderr, sys.stderr))

        # Start both threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to complete
        return_code = process.wait()

        # Wait for both threads to finish reading
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


def task1():
    non_mpi_env, mpi_env = split_mpi_env()
    assert non_mpi_env
    assert mpi_env


@pytest.mark.cpu_only
def test_split_mpi_env():
    session = MpiPoolSession(n_workers=4)
    session.submit_sync(task1)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "task_script", ["_run_mpi_comm_task.py", "_run_multi_mpi_comm_tasks.py"])
def test_llmapi_launch_multiple_tasks(task_script: str):
    """
    Test that the trtllm-llmapi-launch can run multiple tasks.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(cur_dir, task_script)
    assert os.path.exists(test_file), f"Test file {test_file} does not exist"
    command = [
        "mpirun", "-n", "2", "--allow-run-as-root", "trtllm-llmapi-launch",
        "python3", test_file
    ]
    print(' '.join(command))

    with Popen(command,
               env=os.environ,
               stdout=PIPE,
               stderr=PIPE,
               bufsize=1,
               start_new_session=True,
               universal_newlines=True,
               cwd=os.path.dirname(os.path.abspath(__file__))) as process:
        # Function to read from a stream and write to output
        def read_stream(stream, output_stream):
            for line in stream:
                output_stream.write(line)
                output_stream.flush()

        # Create threads to read stdout and stderr concurrently
        stdout_thread = threading.Thread(target=read_stream,
                                         args=(process.stdout, sys.stdout))
        stderr_thread = threading.Thread(target=read_stream,
                                         args=(process.stderr, sys.stderr))

        # Start both threads
        stdout_thread.start()
        stderr_thread.start()

        # Wait for the process to complete
        return_code = process.wait()

        # Wait for both threads to finish reading
        stdout_thread.join()
        stderr_thread.join()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)


# ---- wait_shutdown: shutdown blocks until worker processes actually exit ----


def _wait_workers_exit(identities, timeout: float) -> None:
    """Call the unbound method on an inert stand-in (no MPI spawn).

    ``_wait_workers_exit`` only reads ``self._worker_identities``; a real
    ``MpiPoolSession`` shell would trigger the base class's abort machinery
    at garbage collection.
    """
    import types

    stand_in = types.SimpleNamespace(_worker_identities=identities)
    MpiPoolSession._wait_workers_exit(stand_in, timeout=timeout)


def test_process_start_time_live_and_gone():
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    assert _process_start_time(os.getpid()) is not None
    child = Popen(["true"])  # nosec B603, B607
    child.wait()
    assert _process_start_time(child.pid) is None  # reaped: /proc entry gone


def test_wait_workers_exit_returns_once_workers_are_gone():
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    child = Popen(["true"])  # nosec B603, B607
    identity = (child.pid, _process_start_time(child.pid))
    child.wait()
    # Dead worker -> returns immediately; a None start_time is skipped
    # (identity collection failed for that worker: nothing to wait on).
    _wait_workers_exit((identity, (os.getpid(), None)), timeout=5.0)


def test_wait_workers_exit_bounded_by_timeout_on_live_worker():
    import time as _time

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    t0 = _time.monotonic()
    _wait_workers_exit((me, ), timeout=0.2)  # this process will not exit
    waited = _time.monotonic() - t0
    assert 0.2 <= waited < 2.0  # bounded: a wedged worker cannot hang teardown


def _resolved_futures(values):
    from concurrent.futures import Future

    out = []
    for value in values:
        f = Future()
        f.set_result(value)
        out.append(f)
    return out


def _collect_identities(monkeypatch,
                        results,
                        pending=0,
                        n_workers=2,
                        observed_timeouts=None,
                        bootstrap=None,
                        manager_alive=True,
                        killed=None,
                        bootstrap_delays=None,
                        barrier_delays=None,
                        submit_error=None,
                        result_error=None):
    """Drive _collect_worker_identities on an inert stand-in (no MPI spawn).

    ``results``/``pending`` describe what the *barrier* phase returns.
    ``bootstrap`` describes what the warm-up phase returns; by default the
    warm-up mirrors ``results``, i.e. a pool that bootstrapped normally. Pass
    ``bootstrap=[]`` for a pool whose workers never came up at all, or
    ``bootstrap_delays=[...]`` (seconds, per probe) for a pool whose ranks
    finish ``import tensorrt_llm`` at different times.

    ``barrier_delays`` (seconds, per barrier task) makes the barrier futures
    resolve on a timer measured from *helper entry*, i.e. on the same clock as
    ``bootstrap_delays``. That is what lets a test distinguish a bootstrap that
    waited for every probe from one that returned on the first: only the former
    reaches the barrier late enough to find its futures already done.

    ``submit_error`` makes the *barrier* submit raise; ``result_error`` makes a
    barrier future complete *with* that exception, so ``f.result()`` raises.
    Both stand in for what mpi4py surfaces from a broken pool.
    """
    import threading
    import types
    from concurrent.futures import Future

    from tensorrt_llm.llmapi import mpi_session as m

    if barrier_delays is not None:
        barrier_futs = [Future() for _ in barrier_delays]
        for fut, (delay, value) in zip(barrier_futs,
                                       zip(barrier_delays, results)):
            timer = threading.Timer(delay, fut.set_result, args=(value, ))
            timer.daemon = True
            timer.start()
    else:
        barrier_futs = _resolved_futures(results) + [
            Future() for _ in range(pending)
        ]
    if result_error is not None:
        broken = Future()
        broken.set_exception(result_error)
        barrier_futs = [broken] + barrier_futs[1:]
    if bootstrap is None:
        # Default: the pool bootstrapped normally, i.e. EVERY probe came back.
        # (Padding with unresolved futures here would silently make every
        # caller a partial-bootstrap test.)
        bootstrap = list(results)
        bootstrap += [(9000 + i, b"warm")
                      for i in range(n_workers - len(bootstrap))]
    if bootstrap_delays is not None:
        # Ranks come home on a timer, the way a real staggered bootstrap does.
        hello_futs = [Future() for _ in bootstrap_delays]
        for fut, (delay, value) in zip(hello_futs,
                                       zip(bootstrap_delays, bootstrap)):
            timer = threading.Timer(delay, fut.set_result, args=(value, ))
            timer.daemon = True
            timer.start()
    else:
        hello_futs = _resolved_futures(bootstrap)
        hello_futs += [Future() for _ in range(n_workers - len(hello_futs))]

    real_wait = m.futures_wait

    def _wait(fs, timeout=None, **kwargs):
        if observed_timeouts is not None:
            observed_timeouts.append(timeout)
        return real_wait(fs, timeout=timeout, **kwargs)

    monkeypatch.setattr(m, "futures_wait", _wait)
    # Keep the bootstrap phase short: these tests exercise the control flow,
    # not the production budget (which is asserted directly elsewhere).
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT",
                       os.environ.get("TRTLLM_MPI_IDENTITY_TIMEOUT", "0.3"))
    monkeypatch.setattr(m, "_BOOTSTRAP_POLL_INTERVAL", 0.05)
    monkeypatch.setattr(m, "_IDENTITY_BARRIER_TIMEOUT", 0.1)

    if killed is None:
        killed = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))
    queues = {
        m._worker_hello: iter(hello_futs),
        m._worker_identity_barrier: iter(barrier_futs),
    }

    def _submit(fn):
        if submit_error is not None and fn is m._worker_identity_barrier:
            raise submit_error
        return next(queues[fn])

    pool_closed = []
    stand_in = types.SimpleNamespace(
        n_workers=n_workers,
        mpi_pool=types.SimpleNamespace(
            submit=_submit,
            shutdown=lambda wait=True: pool_closed.append(wait)),
    )
    stand_in._pool_closed = pool_closed
    stand_in._teardown_unidentified_pool = (
        lambda ids: MpiPoolSession._teardown_unidentified_pool(stand_in, ids))
    stand_in._pool_can_make_progress = lambda: manager_alive
    stand_in._wait_worker_bootstrap = (
        lambda futures, timeout: MpiPoolSession._wait_worker_bootstrap(
            stand_in, futures, timeout))
    try:
        result = MpiPoolSession._collect_worker_identities(stand_in)
    except BaseException as e:
        # Expose the stand-in so teardown assertions survive the raise.
        e._stand_in = stand_in
        raise
    return result, killed


def test_identity_collection_complete_returns_identities(monkeypatch):
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    other = (1, b"1")  # pid 1: exists but start_time won't match -> unique pid
    ids, killed = _collect_identities(monkeypatch, [me, other])
    assert set(ids) == {me, other} and not killed


def test_identity_collection_fails_closed_on_timeout(monkeypatch):
    # A pending barrier task means the pool cannot honor wait_shutdown:
    # the session must be torn down and rejected, NOT handed out with the
    # contract silently downgraded (review requirement).
    import pytest as _pytest

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    with _pytest.raises(RuntimeError, match="incomplete"):
        _collect_identities(monkeypatch, [me], pending=1)


def test_identity_gate_marks_every_failure_path(monkeypatch):
    """(d) One greppable marker, on every way this gate can fail.

    The marker lives in the exception message because that is the channel
    proven to reach CI logs for this code path.
    """
    import pytest as _pytest

    from tensorrt_llm.llmapi.mpi_session import (_IDENTITY_GATE_MARKER,
                                                 _process_start_time)

    me = (os.getpid(), _process_start_time(os.getpid()))
    # barrier phase: bootstrapped, then a worker never reached the barrier
    with _pytest.raises(RuntimeError) as barrier_exc:
        _collect_identities(monkeypatch, [me], pending=1)
    # bootstrap phase: no worker ever ran a task
    with _pytest.raises(RuntimeError) as bootstrap_exc:
        _collect_identities(monkeypatch, [me], pending=1, bootstrap=[])

    assert _IDENTITY_GATE_MARKER in str(barrier_exc.value)
    assert _IDENTITY_GATE_MARKER in str(bootstrap_exc.value)
    # ...and the two phases are distinguishable, which is the whole point:
    # a dead pool and a slow one used to emit byte-identical text.
    assert "phase=barrier" in str(barrier_exc.value)
    assert "phase=bootstrap" in str(bootstrap_exc.value)


def test_dead_pool_fails_without_burning_the_bootstrap_budget(monkeypatch):
    """Blocker 1: dead and slow must not cost the same wall time.

    mpi4py raises a failed ``MPI_Comm_spawn`` inside its manager thread, so
    the futures simply never resolve. The thread's death is the positive
    signal that separates that from a bootstrap that is merely slow.
    """
    import time as _time

    import pytest as _pytest

    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", "30")
    t0 = _time.monotonic()
    with _pytest.raises(RuntimeError, match="phase=bootstrap"):
        _collect_identities(monkeypatch, [],
                            pending=2,
                            bootstrap=[],
                            manager_alive=False)
    elapsed = _time.monotonic() - t0
    # Gives up on the positive death signal, not on the 30s deadline.
    assert elapsed < 5.0


def test_slow_bootstrap_is_not_mistaken_for_a_dead_pool(monkeypatch):
    """The other half: a live-but-slow pool must be waited for, not killed.

    The probes land late and out of order, as a real staggered bootstrap
    does — the gate must sit through it and still succeed.
    """
    import time as _time

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    other = (1, b"1")
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", "10")
    t0 = _time.monotonic()
    ids, killed = _collect_identities(monkeypatch, [me, other],
                                      bootstrap=[me, other],
                                      bootstrap_delays=[0.05, 0.6])
    waited = _time.monotonic() - t0
    assert set(ids) == {me, other} and not killed
    # It genuinely waited for the straggler rather than returning on the first.
    assert 0.6 <= waited < 5.0


def test_bootstrap_waits_for_every_worker_not_just_the_first(monkeypatch):
    """A healthy pool with bootstrap skew must not be torn down.

    Returning as soon as ONE probe lands would hand the remaining ranks'
    import time to the barrier's tight fixed budget, converting an absolute
    bound into a skew bound — and tearing down pools that are merely uneven.
    Here the straggler trails the leader by far more than the barrier budget.
    """
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    other = (1, b"1")
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", "10")
    # The barrier results land on the SAME clock as the straggler's probe. A
    # bootstrap that returns on the first probe reaches the barrier at ~t=0 and
    # gets only its 0.1s budget, so the t=0.8s barrier result is still missing
    # and the pool is torn down; a bootstrap that waits for every probe reaches
    # it at t=0.8 and finds the results already there. Without this coupling
    # the test passes either way, which is no guarantee at all.
    ids, killed = _collect_identities(monkeypatch, [me, other],
                                      bootstrap=[me, other],
                                      bootstrap_delays=[0.0, 0.8],
                                      barrier_delays=[0.0, 0.8])
    assert set(ids) == {me, other} and not killed


def test_gate_tears_down_on_runtime_errors_from_the_pool(monkeypatch):
    """mpi4py.MPI.Exception and BrokenExecutor are both RuntimeErrors.

    Both come out of the gate's own submit()/result() calls. If the gate
    re-raised on the RuntimeError base class it would let them past the
    teardown, leaving mpi_pool dangling on a half-constructed session whose
    __del__ then blocks on a broken pool — a hang, in the change that exists
    to remove hangs.

    Covers both arrival routes: raised out of submit(), and delivered through
    a future so f.result() re-raises it.
    """
    from concurrent.futures import BrokenExecutor

    import pytest as _pytest

    from tensorrt_llm.llmapi.mpi_session import (_IDENTITY_GATE_MARKER,
                                                 _process_start_time)

    me = (os.getpid(), _process_start_time(os.getpid()))
    errors = [
        BrokenExecutor("pool is broken"),
        RuntimeError("plain runtime error"),
    ]
    mpi = pytest.importorskip("mpi4py.MPI")
    errors.append(mpi.Exception(mpi.ERR_COMM))

    for how in ("submit", "result"):
        for error in errors:
            label = f"{type(error).__name__} via {how}"
            with _pytest.raises(RuntimeError) as exc:
                _collect_identities(monkeypatch, [me],
                                    n_workers=1,
                                    bootstrap=[me],
                                    **{f"{how}_error": error})
            stand_in = exc.value._stand_in
            # Torn down: pool disconnected without waiting, and cleared.
            assert stand_in._pool_closed == [
                False
            ], (f"{label} escaped the teardown")
            assert stand_in.mpi_pool is None, label
            assert _IDENTITY_GATE_MARKER in str(exc.value), label
            # It failed during the BARRIER, so it must not blame bootstrap and
            # point on-call at TRTLLM_MPI_IDENTITY_TIMEOUT, which is powerless
            # against a broken executor.
            assert "phase=barrier" in str(exc.value), label
            assert "TRTLLM_MPI_IDENTITY_TIMEOUT" not in str(exc.value), label


def test_mpi_and_executor_errors_really_are_runtime_errors():
    """Pins the premise of the test above; if it ever stops holding, say so."""
    from concurrent.futures import BrokenExecutor

    assert issubclass(BrokenExecutor, RuntimeError)
    mpi4py = pytest.importorskip("mpi4py.MPI")
    assert issubclass(mpi4py.Exception, RuntimeError)


def test_pool_can_make_progress_reads_the_real_mpi4py_internals():
    """M1: exercise the real predicate, not a stub.

    It is a getattr chain, so a renamed mpi4py internal would silently make it
    return True forever and the fast-fail would quietly degrade to the full
    deadline with every other test still green.
    """
    import threading
    import types

    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    def probe(pool):
        return MpiPoolSession._pool_can_make_progress(
            types.SimpleNamespace(mpi_pool=types.SimpleNamespace(_pool=pool)))

    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    assert probe(types.SimpleNamespace(thread=dead)) is False

    alive = threading.Event()
    running = threading.Thread(target=alive.wait, daemon=True)
    running.start()
    try:
        assert probe(types.SimpleNamespace(thread=running)) is True
    finally:
        alive.set()
        running.join(timeout=5)

    # Fail-safe: unknown shape must not be reported as dead.
    assert probe(None) is True
    assert probe(types.SimpleNamespace()) is True


def test_mpi4py_still_exposes_the_liveness_handle():
    """Fail loudly on an mpi4py bump that moves the manager thread.

    ``_pool_can_make_progress`` walks ``executor._pool.thread`` through
    ``getattr(..., None)`` and reports "can still progress" on any miss —
    deliberately, so an unrecognised internal degrades to the deadline instead
    of killing healthy pools. The cost is that a rename is invisible at
    runtime, so assert every link of that chain here.
    """
    pytest.importorskip("mpi4py.futures")
    from mpi4py.futures import MPIPoolExecutor

    assert hasattr(MPIPoolExecutor,
                   "_bootstrap"), ("mpi4py no longer bootstraps a pool lazily; "
                                   "_pool_can_make_progress needs revisiting")
    from mpi4py.futures import _lib

    degraded = ("the identity gate's dead-pool fast-fail has silently "
                "degraded to the full deadline (it fails safe, so nothing "
                "else will tell you)")
    # Link 1: the executor stores its pool on `_pool`.
    assert "_pool" in MPIPoolExecutor._bootstrap.__code__.co_names, (
        f"mpi4py's executor no longer keeps its pool on `_pool`; {degraded}")
    # Link 2: the pool keeps its manager on `thread`...
    init_names = _lib.Pool.__init__.__code__.co_names
    assert "thread" in init_names, (
        f"mpi4py's Pool no longer keeps a manager thread; {degraded}")
    # ...and that manager is a started threading.Thread, so `.is_alive()`
    # still means what the predicate assumes it means.
    assert "Thread" in init_names and "start" in init_names, (
        f"mpi4py's Pool manager is no longer a started threading.Thread; "
        f"`.is_alive()` may no longer signal a dead pool. {degraded}")


def test_barrier_phase_waits_on_the_barrier_budget_not_the_bootstrap_one(
        monkeypatch):
    """The consequential wiring: which number reaches futures_wait.

    If the barrier phase ever picked up the bootstrap budget, the guard would
    silently become a 300s one again and every value-level assertion here
    would still pass. So assert the deadline actually passed to the wait.
    """
    from tensorrt_llm.llmapi import mpi_session as m
    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    other = (1, b"1")
    observed = []
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", "37")
    _collect_identities(monkeypatch, [me, other], observed_timeouts=observed)

    # Last wait is the barrier; it uses the fixed barrier budget (pinned to
    # 0.1 by the helper), never the 37s bootstrap budget.
    assert observed[-1] == m._IDENTITY_BARRIER_TIMEOUT
    assert 37 not in observed
    # Every bootstrap wait is a poll slice, never one long blocking wait.
    assert all(t <= m._BOOTSTRAP_POLL_INTERVAL for t in observed[:-1])


def test_barrier_deadline_is_tight_and_untunable(monkeypatch):
    """The guard's own bound must not inherit the bootstrap budget."""
    from tensorrt_llm.llmapi import mpi_session as m

    assert m._IDENTITY_BARRIER_TIMEOUT == 60.0
    # ...and it is strictly tighter than the bootstrap budget it replaced,
    # so widening the bootstrap budget never widens the guard.
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", "9999")
    assert m._IDENTITY_BARRIER_TIMEOUT == 60.0
    monkeypatch.delenv("TRTLLM_MPI_IDENTITY_TIMEOUT", raising=False)
    assert m._IDENTITY_BARRIER_TIMEOUT < m._identity_bootstrap_timeout()
    assert (m.identity_gate_budget() == m._identity_bootstrap_timeout() +
            m._IDENTITY_BARRIER_TIMEOUT)


def test_warm_up_identities_are_reaped_when_the_barrier_stalls(monkeypatch):
    """Blocker 2's root cause: a 0/N failure used to reap nothing.

    The warm-up probes are collective-free, so they come back even when the
    barrier does not — giving the teardown real PIDs to SIGKILL instead of an
    empty tuple.
    """
    import pytest as _pytest

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    killed = []
    with _pytest.raises(RuntimeError, match="phase=barrier"):
        _collect_identities(monkeypatch, [],
                            pending=1,
                            n_workers=1,
                            bootstrap=[me],
                            killed=killed)
    # Before warm-then-measure the barrier returned nothing, partial_identities
    # was empty, and the SIGKILL loop was a no-op.
    assert killed == [os.getpid()]


def test_identity_collection_fails_closed_on_duplicate_pids(monkeypatch):
    import pytest as _pytest

    from tensorrt_llm.llmapi.mpi_session import _process_start_time

    me = (os.getpid(), _process_start_time(os.getpid()))
    with _pytest.raises(RuntimeError, match="incomplete"):
        _collect_identities(monkeypatch, [me, me])  # one worker answered twice


def test_identity_collection_uses_configured_timeout(monkeypatch):
    # The env override now sizes the BOOTSTRAP phase (spawn + import), which
    # is the phase whose cost varies by node. The barrier phase keeps its own
    # fixed, tight deadline.
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", "123.5")
    assert _identity_bootstrap_timeout() == 123.5


def test_identity_timeout_covers_worker_bootstrap(monkeypatch):
    # The deadline bounds spawn + `import tensorrt_llm`, not barrier latency:
    # it must exceed the slowest bootstrap the repo measures (~117s busy node).
    monkeypatch.delenv("TRTLLM_MPI_IDENTITY_TIMEOUT", raising=False)
    assert _identity_bootstrap_timeout() > 117.0


def test_wait_shutdown_false_never_touches_the_identity_gate(monkeypatch):
    """The production default must be byte-for-byte unaffected by this gate.

    ``llm.py`` constructs ``MpiPoolSession`` without ``wait_shutdown``; the
    two-phase gate (and its warm-up submits) must never run on that path.
    """
    import types

    from tensorrt_llm.llmapi import mpi_session as m

    submitted = []
    monkeypatch.setattr(
        m.MpiPoolSession, "_start_mpi_pool", lambda self: setattr(
            self, "mpi_pool",
            types.SimpleNamespace(submit=lambda fn: submitted.append(fn))))
    monkeypatch.setattr(
        m.MpiPoolSession, "_collect_worker_identities",
        lambda self: pytest.fail("identity gate ran with wait_shutdown=False"))
    monkeypatch.setattr(m, "ENABLE_MULTI_DEVICE", False)
    # The inert pool cannot be joined; keep __del__ from arming the abort
    # watchdog on it.
    monkeypatch.setattr(m.MpiPoolSession, "shutdown_abort", lambda self: None)

    session = m.MpiPoolSession(n_workers=2)
    assert session._worker_identities == ()
    assert submitted == []  # no warm-up probe, no barrier: nothing submitted


# Invalid values (unparsable, non-positive) fall back to the default rather
# than turning the barrier into a busy-wait or an unbounded block.
@pytest.mark.parametrize("raw, expected",
                         [("90", 90.0), ("0.5", 0.5),
                          ("", _DEFAULT_IDENTITY_TIMEOUT),
                          ("0", _DEFAULT_IDENTITY_TIMEOUT),
                          ("-1", _DEFAULT_IDENTITY_TIMEOUT),
                          ("abc", _DEFAULT_IDENTITY_TIMEOUT),
                          ("nan", _DEFAULT_IDENTITY_TIMEOUT),
                          ("inf", _DEFAULT_IDENTITY_TIMEOUT),
                          ("-inf", _DEFAULT_IDENTITY_TIMEOUT),
                          ("1e309", _DEFAULT_IDENTITY_TIMEOUT)])
def test_identity_timeout_env_override(monkeypatch, raw, expected):
    monkeypatch.setenv("TRTLLM_MPI_IDENTITY_TIMEOUT", raw)
    assert _identity_bootstrap_timeout() == expected


def test_prefetch_fallback_identity_timeout_matches_mpi_default():
    from test_common.session_prefetcher import _FALLBACK_IDENTITY_TIMEOUT

    assert _FALLBACK_IDENTITY_TIMEOUT == _DEFAULT_IDENTITY_TIMEOUT
