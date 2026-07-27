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

from tensorrt_llm.executor.base_worker import AwaitResponseHelper
from tensorrt_llm.executor.utils import ErrorResponse


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

    def __init__(self, engine, num_pending: int = 1):
        self.engine = engine
        self._results = {cid: _ResultStub() for cid in range(1, num_pending + 1)}
        self.popped = []
        self.result_queue = None
        self.postproc_queues = None

        # Echoed straight back so __call__'s filter is a no-op.
        def _engine_response_callback(r):
            return r

        self._engine_response_callback = _engine_response_callback

    def return_queue(self, client_id: int):
        return self._results[client_id].queue

    def _pop_result(self, client_id: int):
        self.popped.append(client_id)
        self._results.pop(client_id, None)


def _make_helper(engine, num_pending: int = 1):
    helper = AwaitResponseHelper.__new__(AwaitResponseHelper)
    helper.worker = _WorkerStub(engine, num_pending=num_pending)
    helper.handler_kind = AwaitResponseHelper.HandlerKind.unknown
    helper.enable_postprocprocess_parallel = False
    helper.temp_error_responses = _stdlib_queue.Queue()
    return helper


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
