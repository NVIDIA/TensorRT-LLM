"""Unit tests for AwaitResponseHelper.__call__ event-loop crash handling.

When PyExecutor's event-loop thread dies (e.g. KV cache OOM), every
pending ``GenerationResult`` parked in ``queue.get()`` / ``aqueue.get()``
must wake up with a meaningful ``ErrorResponse`` rather than hang
forever. See nvbug 6038228 and PR #12735.

These tests bind the real ``AwaitResponseHelper.__call__`` /
``_broadcast_event_loop_error`` to lightweight stubs, so they need
neither GPUs nor models.
"""

import datetime
import queue as _stdlib_queue
import threading

import pytest

from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.utils import ErrorResponse, RequestError
from tensorrt_llm.executor.worker import GenerationExecutorWorker

pytestmark = pytest.mark.cpu_only


class _EngineStub:
    """Stub for self.worker.engine: returns whatever the test plugged in."""

    def __init__(
        self,
        await_responses_result=None,
        await_responses_raises=None,
        event_loop_error=None,
        is_shutdown=False,
    ):
        self._await_responses_result = await_responses_result or []
        self._await_responses_raises = await_responses_raises
        self._event_loop_error = event_loop_error
        self.is_shutdown = is_shutdown
        self.calls = 0
        # The rank-crash kill's delivery gate. Real PyExecutor creates this in
        # __init__; the helper only ever sets it, never reads it.
        self._event_loop_error_delivered = threading.Event()

    def await_responses(self, timeout: datetime.timedelta):
        self.calls += 1
        if self._await_responses_raises is not None:
            raise self._await_responses_raises
        return list(self._await_responses_result)


class _ResultStub:
    """Minimal GenerationResult: just exposes a queue."""

    def __init__(self):
        self.queue = _stdlib_queue.Queue()


class _WorkerStub:
    """Stub for BaseWorker exposing only the attributes the helper touches."""

    def __init__(self, engine, num_pending: int = 1, ipc: bool = False):
        self.engine = engine
        self._results = {cid: _ResultStub() for cid in range(1, num_pending + 1)}
        self.popped = []
        # A non-None result_queue is what makes the helper resolve to
        # ipc_batched -- i.e. the proxy/spawned-worker deployment.
        self.result_queue = object() if ipc else None
        self.postproc_queues = None
        # responses_handler() reads this unguarded (base_worker.py); BaseWorker
        # sets it in __init__, which this stub bypasses, so seed it to None.
        self.frontend_result_queues = None

        # Echoed straight back so __call__'s filter is a no-op.
        def _engine_response_callback(r):
            return r

        self._engine_response_callback = _engine_response_callback

    def return_queue(self, client_id: int):
        return self._results[client_id].queue

    def _pop_result(self, client_id: int):
        self.popped.append(client_id)
        self._results.pop(client_id, None)


def _make_helper(engine, num_pending: int = 1, ipc: bool = False):
    helper = AwaitResponseHelper.__new__(AwaitResponseHelper)
    helper.worker = _WorkerStub(engine, num_pending=num_pending, ipc=ipc)
    helper.handler_kind = AwaitResponseHelper.HandlerKind.unknown
    helper.enable_postprocprocess_parallel = False
    helper.temp_error_responses = _stdlib_queue.Queue()
    return helper


class _ThreadStub:
    """ManagedThread stand-in: ident set + not alive means "already exited"."""

    def __init__(self, *, alive=False, ident=1):
        self._alive = alive
        self.ident = ident
        self.starts = 0

    def is_alive(self):
        return self._alive

    def start(self):
        self.starts += 1


class _StartThreadWorkerStub:
    """Minimal stand-in for the worker start_thread() binds to.

    It only reads self.engine, so a plain object avoids an uninitialized
    GenerationExecutorWorker whose destructor would raise at collection time.
    """

    def __init__(self, event_loop_error=None, can_enqueue=True):
        self.engine = _EngineStub(event_loop_error=event_loop_error)
        self.engine.can_enqueue_requests = lambda: can_enqueue


class TestStartThreadAfterExit:
    """start_thread must surface an engine crash instead of restarting."""

    def test_surfaces_engine_error_as_request_error(self):
        original = RuntimeError("kv cache OOM")
        worker = _StartThreadWorkerStub(event_loop_error=original)
        thread = _ThreadStub()

        with pytest.raises(RequestError) as excinfo:
            GenerationExecutorWorker.start_thread(worker, thread)

        # Chained, not re-raised: the caller still sees the real cause.
        assert excinfo.value.__cause__ is original
        assert "kv cache OOM" in str(excinfo.value)
        assert thread.starts == 0

    def test_repeated_calls_do_not_accumulate_traceback(self):
        # start() runs on every submit(); re-raising the same object would grow
        # its __traceback__ one frame per call.
        original = RuntimeError("kv cache OOM")
        worker = _StartThreadWorkerStub(event_loop_error=original)
        thread = _ThreadStub()

        raised = []
        for _ in range(3):
            with pytest.raises(RequestError) as excinfo:
                GenerationExecutorWorker.start_thread(worker, thread)
            raised.append(excinfo.value)

        assert len({id(e) for e in raised}) == 3
        assert all(e.__cause__ is original for e in raised)

    def test_post_shutdown_exit_returns_quietly(self):
        # The other exit path: shutdown() called ManagedThread.stop(), so
        # stop_event ended run() and there is no error to report.
        worker = _StartThreadWorkerStub(event_loop_error=None)
        thread = _ThreadStub()

        GenerationExecutorWorker.start_thread(worker, thread)

        assert thread.starts == 0

    def test_fresh_thread_is_started(self):
        worker = _StartThreadWorkerStub(event_loop_error=None)
        thread = _ThreadStub(ident=None)

        GenerationExecutorWorker.start_thread(worker, thread)

        assert thread.starts == 1

    def test_does_not_start_when_enqueueing_is_disabled(self):
        # The can_enqueue_requests() guard returns before the error check, so a
        # stashed error must not surface either.
        worker = _StartThreadWorkerStub(
            event_loop_error=RuntimeError("should not surface"), can_enqueue=False
        )
        thread = _ThreadStub(ident=None)

        GenerationExecutorWorker.start_thread(worker, thread)

        assert thread.starts == 0


class TestAwaitResponseHelperEventLoopError:
    def test_normal_path_returns_true(self):
        """No engine error and no responses: ManagedThread should keep going."""
        engine = _EngineStub(await_responses_result=[])
        helper = _make_helper(engine, num_pending=1)

        assert helper(timeout=0.01) is True
        # No ErrorResponse should have been pushed.
        for rs in helper.worker._results.values():
            assert rs.queue.empty()
        assert helper.worker.popped == []

    def test_broadcasts_when_await_responses_raises(self):
        """Defensive: any exception out of engine.await_responses triggers broadcast."""
        original = RuntimeError("Event loop terminated with error: KV OOM")
        engine = _EngineStub(await_responses_raises=original)
        helper = _make_helper(engine, num_pending=2)

        assert helper(timeout=0.01) is False  # ManagedThread should stop

        # Each pending GenerationResult got an ErrorResponse.
        for cid in (1, 2):
            err = helper.worker._results.get(cid)
            assert err is None, "result should have been popped"
        # popped order is iteration order over dict keys (insertion order in py3.7+)
        assert sorted(helper.worker.popped) == [1, 2]

    def test_broadcasts_when_event_loop_error_set_after_empty_response(self):
        """Broadcast must fire even when await_responses returns [] silently.

        ``_await_any_response`` returns ``[]`` on shutdown without raising,
        but ``_event_loop_error`` is still stashed on the engine.
        """
        original = RuntimeError("KV cache OOM")
        engine = _EngineStub(await_responses_result=[], event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=3)

        assert helper(timeout=0.01) is False
        assert sorted(helper.worker.popped) == [1, 2, 3]

    def test_pushed_response_is_error_response_with_message(self):
        """Pushed item is an ErrorResponse carrying the original error text."""
        original = RuntimeError("KV cache OOM at iteration 42")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        # Capture queue refs before they get popped from _results.
        helper = _make_helper(engine, num_pending=1)
        result_queue = helper.worker.return_queue(client_id=1)

        helper(timeout=0.01)

        item = result_queue.get_nowait()
        assert isinstance(item, ErrorResponse)
        assert item.client_id == 1
        assert "KV cache OOM" in item.error_msg
        assert "Event loop terminated" in item.error_msg

    def test_no_pending_results_returns_false_quietly(self):
        """Crash with no pending requests still stops the thread cleanly."""
        original = RuntimeError("crash")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=0)

        assert helper(timeout=0.01) is False
        assert helper.worker.popped == []

    def test_broadcast_helper_idempotent_via_pop(self):
        """Calling _broadcast_event_loop_error twice is safe (second is a no-op)."""
        original = RuntimeError("crash")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=2)

        assert helper._broadcast_event_loop_error(original) is False
        assert sorted(helper.worker.popped) == [1, 2]
        # second time around: nothing left to wake.
        assert helper._broadcast_event_loop_error(original) is False


class TestEventLoopErrorDeliveryGate:
    """The gate that stands the rank-crash hard kill down.

    Setting it means "a client is holding the real error, so killing the
    world would only replace a traceback with exit 137". It may therefore
    only be set when a client verifiably woke. Setting it optimistically on
    the proxy/IPC path -- the default when the LLM spawns MPI workers --
    disarms the 10s kill while the peer ranks are still stranded, regressing
    exactly the case the kill exists for back to the 300s HangDetector.
    """

    def test_gate_set_when_client_woken_in_single_process_mode(self):
        original = RuntimeError("KV cache OOM")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=2, ipc=False)

        assert helper(timeout=0.01) is False
        assert helper.handler_kind is AwaitResponseHelper.HandlerKind.single_process_worker
        assert engine._event_loop_error_delivered.is_set()

    def test_gate_stays_clear_on_ipc_batched_path(self):
        """The regression this class exists for.

        On ``ipc_batched`` the worker-side ``_results`` queues written by the
        broadcast have no reader -- responses travel via
        ``handle_for_ipc_batched`` -- so nothing reached the client even
        though the puts succeeded.
        """
        original = RuntimeError("KV cache OOM")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=2, ipc=True)

        assert helper(timeout=0.01) is False
        assert helper.handler_kind is AwaitResponseHelper.HandlerKind.ipc_batched
        assert not engine._event_loop_error_delivered.is_set()

    def test_gate_stays_clear_when_there_was_nobody_to_wake(self):
        """No pending request means nobody is holding the error.

        So the kill must stay armed even in single-process mode.
        """
        original = RuntimeError("crash")
        engine = _EngineStub(event_loop_error=original, is_shutdown=True)
        helper = _make_helper(engine, num_pending=0, ipc=False)

        assert helper(timeout=0.01) is False
        assert not engine._event_loop_error_delivered.is_set()

    def test_gate_set_on_the_await_responses_raises_path_too(self):
        """The defensive branch also delivers, so it may stand the kill down.

        Previously it never set the gate at all.
        """
        original = RuntimeError("unexpected")
        engine = _EngineStub(await_responses_raises=original)
        helper = _make_helper(engine, num_pending=1, ipc=False)

        assert helper(timeout=0.01) is False
        assert engine._event_loop_error_delivered.is_set()
