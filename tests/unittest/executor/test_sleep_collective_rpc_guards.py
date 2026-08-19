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
"""Pure-Python guard tests for MPI sleep/wakeup and collective_rpc.

No GPU or model weights required; all CUDA/MPI/ZMQ machinery is bypassed
via object.__new__ + manual attribute injection.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.cpu_only


# Sentinel used as the default sleep_config value in _make_worker so that
# callers who omit sleep_config get a truthy non-None object (simulating a
# configured SleepConfig), while callers who pass sleep_config=None test the
# "feature not enabled" guard.  Module-level placement avoids Ruff B008.
_SLEEP_CONFIG_DEFAULT = object()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_worker(backend="pytorch", world_size=1, sleep_config=_SLEEP_CONFIG_DEFAULT):
    """Construct a BaseWorker shell without triggering MPI/CUDA init."""
    from tensorrt_llm.executor.base_worker import BaseWorker

    w = object.__new__(BaseWorker)
    w._backend = backend
    w._is_pytorch_backend = backend in ("pytorch", "_autodeploy")
    w.llm_args = SimpleNamespace(
        backend=backend,
        parallel_config=SimpleNamespace(world_size=world_size),
        sleep_config=sleep_config,
    )
    w.engine = SimpleNamespace(
        begin_sleep_transition=MagicMock(),
        complete_sleep_transition=MagicMock(),
        abort_sleep_transition=MagicMock(),
        begin_wakeup_transition=MagicMock(),
        complete_wakeup_transition=MagicMock(),
        abort_wakeup_transition=MagicMock(),
        fail_sleep_wakeup_transition=MagicMock(),
    )
    return w


def _make_proxy(cls_name, model_world_size=1, rpc_client=None):
    """Construct an IPC or RPC proxy shell without triggering ZMQ/MPI init."""
    if cls_name == "ipc":
        from tensorrt_llm.executor.proxy import GenerationExecutorProxy as Cls
    else:
        from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy as Cls
    p = object.__new__(Cls)
    p.model_world_size = model_world_size
    p.rpc_client = rpc_client
    return p


# ---------------------------------------------------------------------------
# BaseWorker.sleep() / wakeup()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sleep", "wakeup"])
class TestBaseWorkerSleepGuards:
    """Guard-path tests for BaseWorker.sleep() and wakeup()."""

    def test_wrong_backend_raises(self, method):
        """Raises ValueError when backend is not 'pytorch'."""
        w = _make_worker(backend="tensorrt")
        with pytest.raises(ValueError, match="only available for the PyTorch"):
            getattr(w, method)(["kv_cache"])

    def test_autodeploy_backend_raises(self, method):
        """AutoDeploy must be excluded.

        Its allocations aren't tagged under sleep_config VMM scopes, so
        release_with_tag would silently no-op instead of freeing GPU memory.
        """
        w = _make_worker(backend="_autodeploy")
        with pytest.raises(ValueError, match="only available for the PyTorch"):
            getattr(w, method)(["kv_cache"])

    def test_missing_sleep_config_raises(self, method):
        """Raises ValueError when sleep_config is not set on llm_args."""
        w = _make_worker(sleep_config=None)
        with pytest.raises(ValueError, match="Sleep feature is not enabled"):
            getattr(w, method)(["kv_cache"])

    def test_multirank_dispatches_to_helper(self, method):
        """world_size > 1 must route to _multi_rank_sleep_wakeup, not raise.

        The precondition check no longer blocks multi-rank; the helper
        handles coordination via the control communicator.
        """
        from unittest.mock import patch

        w = _make_worker(world_size=2)
        with patch.object(w, "_multi_rank_sleep_wakeup") as mock_helper:
            getattr(w, method)(["kv_cache"])
        mock_helper.assert_called_once()
        call_action, call_tags = mock_helper.call_args[0]
        assert call_action == method
        assert len(call_tags) == 1
        getattr(w.engine, f"begin_{method}_transition").assert_called_once_with(call_tags)
        getattr(w.engine, f"complete_{method}_transition").assert_called_once_with()
        getattr(w.engine, f"abort_{method}_transition").assert_not_called()

    def test_multirank_recoverable_failure_restores_admission(self, method):
        """A helper failure before mutation restores the prior admission state."""
        from unittest.mock import patch

        w = _make_worker(world_size=2)
        with (
            patch.object(
                w,
                "_multi_rank_sleep_wakeup",
                side_effect=RuntimeError("prepare failed"),
            ),
            pytest.raises(RuntimeError, match="prepare failed"),
        ):
            getattr(w, method)(["kv_cache"])

        getattr(w.engine, f"abort_{method}_transition").assert_called_once_with()
        getattr(w.engine, f"complete_{method}_transition").assert_not_called()
        w.engine.fail_sleep_wakeup_transition.assert_not_called()

    def test_backend_checked_before_sleep_config(self, method):
        """Backend check fires even when sleep_config is also absent."""
        w = _make_worker(backend="tensorrt", sleep_config=None)
        with pytest.raises(ValueError, match="only available for the PyTorch"):
            getattr(w, method)(["kv_cache"])

    def test_sleep_config_check_fires_for_multirank(self, method):
        """sleep_config check fires even for multi-rank callers.

        The precondition check runs before the world_size dispatch, so a
        multi-rank caller without sleep_config still gets an actionable error.
        """
        w = _make_worker(world_size=2, sleep_config=None)
        with pytest.raises(ValueError, match="Sleep feature is not enabled"):
            getattr(w, method)(["kv_cache"])


# ---------------------------------------------------------------------------
# _multi_rank_sleep_wakeup serialisation via _sleep_wakeup_lock
# ---------------------------------------------------------------------------


class TestMultiRankSleepWakeupLock:
    """Verify that _multi_rank_sleep_wakeup holds engine._sleep_wakeup_lock.

    The lock prevents two concurrent RPC calls from interleaving their
    control_action + _sleep_wakeup_comm send/recv sequences.  We test this by
    spying on the lock's __enter__/__exit__ calls while driving
    _multi_rank_sleep_wakeup directly (bypassing the sleep/wakeup dispatch
    layer), with all MPI/CUDA machinery stubbed out.
    """

    def _make_multi_rank_worker(self):
        """BaseWorker shell wired for a 2-rank deployment."""
        import threading
        from contextlib import contextmanager

        from tensorrt_llm.executor.base_worker import BaseWorker

        w = object.__new__(BaseWorker)
        w._backend = "pytorch"
        w.rank = 0
        w.llm_args = SimpleNamespace(
            backend="pytorch",
            parallel_config=SimpleNamespace(world_size=2),
            sleep_config=object(),
        )

        # A real lock so the actual acquire/release semantics are exercised.
        real_lock = threading.Lock()
        lock_events: list = []

        class SpyLock:
            """Wraps a real Lock and records enter/exit order."""

            def __enter__(self_inner):
                lock_events.append("acquire")
                return real_lock.__enter__()

            def __exit__(self_inner, *args):
                lock_events.append("release")
                return real_lock.__exit__(*args)

        # Stub out _sleep_wakeup_comm; send captures op_id, recv returns ok ACK.
        class MockComm:
            def __init__(self):
                self.op_id = None
                self.phase_by_source = {}

            def send(self, payload, *args, **kwargs):
                self.op_id = payload.get("op_id", self.op_id)
                dest = kwargs.get("dest")
                if dest is not None:
                    self.phase_by_source[dest] = payload.get("action")

            def iprobe(self, *args, **kwargs):
                return True

            def recv(self, *args, **kwargs):
                source = kwargs.get("source")
                return {
                    "status": "ok",
                    "op_id": self.op_id,
                    "phase": self.phase_by_source.get(source),
                }

        mock_comm = MockComm()

        @contextmanager
        def _noop_control_action(**kwargs):
            yield None

        w.engine = SimpleNamespace(
            _sleep_wakeup_lock=SpyLock(),
            _sleep_wakeup_comm=mock_comm,
            control_action=_noop_control_action,
        )

        return w, lock_events

    def test_lock_acquired_and_released(self):
        """_sleep_wakeup_lock must be acquired before and released after.

        Covers the full control_action + send/recv sequence.
        """
        from unittest.mock import patch

        w, lock_events = self._make_multi_rank_worker()

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

            w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        assert lock_events == ["acquire", "release"], (
            f"Expected ['acquire', 'release'], got {lock_events}"
        )

    def test_concurrent_calls_do_not_interleave(self):
        """Two concurrent _multi_rank_sleep_wakeup calls must not overlap.

        We use a threading.Barrier to maximise the chance of a race, then
        check that the critical section (between acquire and release) is
        never entered by both threads simultaneously.

        All CUDA/VMM patches are applied once in the main thread so both
        worker threads share the same mock objects.  Per-thread patch contexts
        for the same globals can restore each other's mocks in the wrong order
        and leave patched state behind for later tests.
        """
        import threading
        from unittest.mock import patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        w, _ = self._make_multi_rank_worker()

        overlap_detected = threading.Event()
        inside = threading.Event()  # set while a thread is in the critical sec.

        def slow_send(*args, **kwargs):
            """Simulate work inside the critical section."""
            payload = args[0]
            w.engine._sleep_wakeup_comm.op_id = payload.get(
                "op_id", w.engine._sleep_wakeup_comm.op_id
            )
            if inside.is_set():
                overlap_detected.set()
            inside.set()
            import time

            time.sleep(0.01)
            inside.clear()

        w.engine._sleep_wakeup_comm.send = slow_send
        barrier = threading.Barrier(2)
        thread_exceptions: list = []

        def run():
            barrier.wait()
            try:
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])
            except Exception as exc:  # noqa: BLE001
                thread_exceptions.append(exc)

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            t1 = threading.Thread(target=run)
            t2 = threading.Thread(target=run)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        if thread_exceptions:
            raise AssertionError(
                f"Worker thread(s) raised unexpected exceptions: {thread_exceptions}"
            )
        assert not overlap_detected.is_set(), (
            "Two _multi_rank_sleep_wakeup calls overlapped in the critical "
            "section; _sleep_wakeup_lock is not being held correctly."
        )


# ---------------------------------------------------------------------------
# _multi_rank_sleep_wakeup protocol error paths
# ---------------------------------------------------------------------------


def _make_proto_worker(recv_responses, world_size=3):
    """Return a BaseWorker shell whose _sleep_wakeup_comm drives recv_responses.

    ``recv_responses`` is a list consumed left-to-right by each recv() call.
    send() is a no-op.  The engine has a real Lock and a no-op control_action.
    ``world_size`` defaults to 3 so tests cover more than one peer ACK.
    """
    import threading
    from contextlib import contextmanager

    from tensorrt_llm.executor.base_worker import BaseWorker

    responses = list(recv_responses)

    w = object.__new__(BaseWorker)
    w._backend = "pytorch"
    w.rank = 0
    w.llm_args = SimpleNamespace(
        backend="pytorch",
        parallel_config=SimpleNamespace(world_size=world_size),
        sleep_config=object(),
    )

    recv_calls: list = []

    class FakeComm:
        def __init__(self):
            self.op_id = None
            self.phase_by_source = {}

        def send(self, payload, *args, **kwargs):
            self.op_id = payload.get("op_id", self.op_id)
            dest = kwargs.get("dest")
            if dest is not None:
                self.phase_by_source[dest] = payload.get("action")

        def iprobe(self, *args, **kwargs):
            return True

        def recv(self, source, tag):
            recv_calls.append(source)
            ack = responses.pop(0) if responses else {"status": "ok"}
            ack.setdefault("op_id", self.op_id)
            ack.setdefault("phase", self.phase_by_source.get(source))
            return ack

    @contextmanager
    def _noop_control_action(**kwargs):
        yield None

    w.engine = SimpleNamespace(
        _sleep_wakeup_lock=threading.Lock(),
        _sleep_wakeup_comm=FakeComm(),
        control_action=_noop_control_action,
    )
    return w, recv_calls


class TestMultiRankAckErrorPropagation:
    """Peer ACK errors must surface in the RuntimeError raised by _multi_rank_sleep_wakeup().

    Covers both single-peer and multi-peer error aggregation.
    """

    def test_peer_error_ack_raises(self):
        """A non-ok ACK from a peer rank must cause RuntimeError with detail."""
        from unittest.mock import patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        # Two peers (world_size=3): rank-1 ok, rank-2 error.
        responses = [
            {"status": "ok"},
            {"status": "error", "error": "rank 2 exploded"},
        ]
        w, _ = _make_proto_worker(responses)

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="rank 2 exploded"):
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

    def test_multiple_peer_errors_aggregated(self):
        """Errors from several peers must all appear in the raised message."""
        from unittest.mock import patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        responses = [
            {"status": "error", "error": "rank 1 OOM"},
            {"status": "error", "error": "rank 2 OOM"},
        ]
        w, _ = _make_proto_worker(responses)

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        msg = str(exc_info.value)
        assert "rank 1 OOM" in msg
        assert "rank 2 OOM" in msg


class TestMnnvlSleepWakeupCoordination:
    """Native MNNVL hooks enter COMMIT collectively and remain tag-scoped."""

    def test_pyexecutor_discovers_shared_native_resource_once(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from tensorrt_llm._torch.modules.fused_moe.communication.base import (
            CheckpointableCommunication,
        )
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _SleepWakeupAction
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        class FutureCheckpointCommunication:
            def __init__(self, resource_key):
                self.resource_key = resource_key
                self.checkpoint_prepare = Mock()
                self.checkpoint_restore = Mock()

            def checkpoint_resource_key(self):
                return self.resource_key

        shared_resource_key = object()
        resources = []
        modules = []
        for _ in range(2):
            resource = FutureCheckpointCommunication(shared_resource_key)
            assert isinstance(resource, CheckpointableCommunication)
            resources.append(resource)
            modules.append(SimpleNamespace(comm=resource))

        executor = object.__new__(PyExecutor)
        executor.model_engine = SimpleNamespace(model=SimpleNamespace(modules=lambda: modules))
        executor.draft_model_engine = None

        tags = [ExecutorMemoryType.MODEL_ENGINE_MAIN]
        assert executor._mnnvl_checkpoint_resources(tags) == [resources[0]]
        executor._run_mnnvl_checkpoint_resources(_SleepWakeupAction.SLEEP, tags)
        resources[0].checkpoint_prepare.assert_called_once_with()
        resources[1].checkpoint_prepare.assert_not_called()
        assert not executor._has_mnnvl_checkpoint_resources([ExecutorMemoryType.KV_CACHE])

    def test_mnnvl_commit_reaches_peer_before_rank_zero_hook(self):
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.py_executor import _SleepWakeupAction
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        worker, _ = _make_proto_worker(
            [
                {"status": "ok", "has_mnnvl_resources": True},
                {"status": "ok"},
            ],
            world_size=2,
        )
        events = []
        original_send = worker.engine._sleep_wakeup_comm.send

        def record_send(payload, *args, **kwargs):
            events.append(("send", payload["action"]))
            return original_send(payload, *args, **kwargs)

        worker.engine._sleep_wakeup_comm.send = record_send
        # Model partitions can differ: rank 0 may own no MoE layer while a
        # peer does. The PREPARE ACK must still select peer-first COMMIT.
        worker.engine._has_mnnvl_checkpoint_resources = lambda tags: False
        worker.engine._run_mnnvl_checkpoint_resources = lambda action, tags: events.append(
            ("local_mnnvl", action)
        )

        with (
            patch(
                "tensorrt_llm._torch.virtual_memory.release_with_tag",
                side_effect=lambda *tags: events.append(("local_vmm", "sleep")),
            ),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.MODEL_ENGINE_MAIN])

        commit_index = events.index(("send", _SleepWakeupAction.COMMIT))
        hook_index = events.index(("local_mnnvl", _SleepWakeupAction.SLEEP))
        vmm_index = events.index(("local_vmm", "sleep"))
        assert commit_index < hook_index < vmm_index

    def test_kv_only_sleep_does_not_run_mnnvl_hook(self):
        from unittest.mock import Mock, patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        worker, _ = _make_proto_worker([{"status": "ok"}, {"status": "ok"}], world_size=2)
        run_mnnvl = Mock()
        worker.engine._has_mnnvl_checkpoint_resources = lambda tags: False
        worker.engine._run_mnnvl_checkpoint_resources = run_mnnvl

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        run_mnnvl.assert_not_called()


class TestMultiRankSendFailureRecovery:
    """Partial rank-0 broadcast failures must not leave peer ACKs undrained."""

    def test_mid_broadcast_send_failure_drains_action_and_abort_acks(self):
        """If one peer received the action before send() fails, rank-0 drains it.

        The peer that did not receive the original action gets a best-effort
        abort message so its executor loop can leave the control barrier.
        """
        import threading
        from contextlib import contextmanager
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.py_executor import _SleepWakeupAction, _SleepWakeupTag
        from tensorrt_llm.executor.base_worker import BaseWorker
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        send_calls = []
        recv_calls = []

        class FakeComm:
            def __init__(self):
                self.op_id = None

            def send(self, payload, dest, tag):
                self.op_id = payload.get("op_id", self.op_id)
                send_calls.append((payload["action"], dest, tag))
                if payload["action"] == _SleepWakeupAction.PREPARE and dest == 2:
                    raise RuntimeError("simulated mid-broadcast send failure")

            def iprobe(self, source, tag):
                return True

            def recv(self, source, tag):
                recv_calls.append((source, tag))
                if source == 1:
                    return {"status": "ok", "op_id": self.op_id}
                return {
                    "status": "ok",
                    "error": None,
                    "reason": "rank 0 aborted sleep/wakeup before local execution",
                    "op_id": self.op_id,
                }

        @contextmanager
        def _noop_control_action(**kwargs):
            yield None

        w = object.__new__(BaseWorker)
        w._backend = "pytorch"
        w.rank = 0
        w.llm_args = SimpleNamespace(
            backend="pytorch",
            parallel_config=SimpleNamespace(world_size=3),
            sleep_config=object(),
        )
        w.engine = SimpleNamespace(
            _sleep_wakeup_lock=threading.Lock(),
            _sleep_wakeup_comm=FakeComm(),
            control_action=_noop_control_action,
        )
        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag") as release,
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        msg = str(exc_info.value)
        assert "simulated mid-broadcast send failure" in msg
        assert release.call_count == 0
        assert send_calls == [
            (_SleepWakeupAction.PREPARE, 1, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.PREPARE, 2, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.ABORT, 1, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.ABORT, 2, _SleepWakeupTag.ACTION),
        ]
        assert recv_calls == [
            (1, _SleepWakeupTag.ACK),
            (1, _SleepWakeupTag.ACK),
            (2, _SleepWakeupTag.ACK),
        ]

    def test_non_rank_call_raises_runtime_error(self):
        """Rank precondition uses a production RuntimeError, not assert."""
        from tensorrt_llm.executor.base_worker import BaseWorker
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        w = object.__new__(BaseWorker)
        w.rank = 1

        with pytest.raises(RuntimeError, match="rank 0"):
            w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

    def test_missing_sleep_wakeup_comm_raises_runtime_error(self):
        """Communicator precondition uses a production RuntimeError, not assert."""
        from tensorrt_llm.executor.base_worker import BaseWorker
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        w = object.__new__(BaseWorker)
        w.rank = 0
        w.engine = SimpleNamespace(_sleep_wakeup_comm=None)

        with pytest.raises(RuntimeError, match="_sleep_wakeup_comm"):
            w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

    def test_action_ack_drain_is_bounded(self, monkeypatch):
        """A missing action ACK must not block _multi_rank_sleep_wakeup forever."""
        import threading
        from contextlib import contextmanager
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor import py_executor
        from tensorrt_llm.executor.base_worker import BaseWorker
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        class FakeComm:
            def send(self, payload, dest, tag):
                pass

            def iprobe(self, source, tag):
                return False

            def recv(self, source, tag):
                raise AssertionError("ACK drain must probe before recv")

        @contextmanager
        def _noop_control_action(**kwargs):
            yield None

        w = object.__new__(BaseWorker)
        w._backend = "pytorch"
        w.rank = 0
        w.llm_args = SimpleNamespace(
            backend="pytorch",
            parallel_config=SimpleNamespace(world_size=2),
            sleep_config=object(),
        )
        w.engine = SimpleNamespace(
            _sleep_wakeup_lock=threading.Lock(),
            _sleep_wakeup_comm=FakeComm(),
            control_action=_noop_control_action,
        )
        fail_stop = MagicMock()
        w._fail_stop_divergent_sleep_wakeup = fail_stop

        monkeypatch.setattr(py_executor, "_SLEEP_WAKEUP_ACK_TIMEOUT_S", 0.0)
        monkeypatch.setattr(py_executor, "_SLEEP_WAKEUP_ACK_POLL_INTERVAL_S", 0.0)

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="timed out waiting"):
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        fail_stop.assert_called_once()

    def test_abort_send_failure_fail_stops_prepared_worker(self):
        """Rank 0 must not resume if a prepared peer cannot be aborted."""
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.py_executor import _SleepWakeupAction
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        worker, _ = _make_proto_worker(
            [{"status": "error", "error": "prepare failed"}],
            world_size=2,
        )
        original_send = worker.engine._sleep_wakeup_comm.send

        def fail_abort(payload, *args, **kwargs):
            original_send(payload, *args, **kwargs)
            if payload["action"] == _SleepWakeupAction.ABORT:
                raise RuntimeError("abort send failed")

        worker.engine._sleep_wakeup_comm.send = fail_abort
        fail_stop = MagicMock()
        worker._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag") as release,
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="abort send failed"):
                worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        release.assert_not_called()
        fail_stop.assert_called_once()

    def test_abort_error_ack_fail_stops_prepared_worker(self):
        """A genuine ABORT handler error leaves peer recovery inconclusive."""
        from unittest.mock import patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        worker, _ = _make_proto_worker(
            [
                {"status": "error", "error": "prepare failed"},
                {"status": "error", "error": "abort handler failed"},
            ],
            world_size=2,
        )
        fail_stop = MagicMock()
        worker._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag") as release,
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="abort handler failed"):
                worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        release.assert_not_called()
        fail_stop.assert_called_once()

    def test_commit_send_failure_aborts_uncommitted_rank(self):
        """A prepared rank that misses COMMIT must receive ABORT to unblock."""
        import threading
        from contextlib import contextmanager
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.py_executor import _SleepWakeupAction, _SleepWakeupTag
        from tensorrt_llm.executor.base_worker import BaseWorker
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        send_calls = []
        recv_calls = []

        class FakeComm:
            def __init__(self):
                self.op_id = None
                self.ack_phases = {}

            def send(self, payload, dest, tag):
                self.op_id = payload.get("op_id", self.op_id)
                send_calls.append((payload["action"], dest, tag))
                self.ack_phases.setdefault(dest, []).append(payload["action"])
                if payload["action"] == _SleepWakeupAction.COMMIT and dest == 2:
                    raise RuntimeError("simulated commit send failure")

            def iprobe(self, source, tag):
                return True

            def recv(self, source, tag):
                recv_calls.append((source, tag))
                return {
                    "status": "ok",
                    "op_id": self.op_id,
                    "phase": self.ack_phases[source].pop(0),
                }

        @contextmanager
        def _noop_control_action(**kwargs):
            yield None

        w = object.__new__(BaseWorker)
        w._backend = "pytorch"
        w.rank = 0
        w.llm_args = SimpleNamespace(
            backend="pytorch",
            parallel_config=SimpleNamespace(world_size=3),
            sleep_config=object(),
        )
        w.engine = SimpleNamespace(
            _sleep_wakeup_lock=threading.Lock(),
            _sleep_wakeup_comm=FakeComm(),
            control_action=_noop_control_action,
        )
        fail_stop = MagicMock()
        w._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag") as release,
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="commit send failure"):
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        release.assert_called_once()
        fail_stop.assert_called_once()
        assert send_calls == [
            (_SleepWakeupAction.PREPARE, 1, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.PREPARE, 2, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.COMMIT, 1, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.COMMIT, 2, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.ABORT, 2, _SleepWakeupTag.ACTION),
        ]
        assert recv_calls == [
            (1, _SleepWakeupTag.ACK),
            (2, _SleepWakeupTag.ACK),
            (2, _SleepWakeupTag.ACK),
            (2, _SleepWakeupTag.ACK),
            (1, _SleepWakeupTag.ACK),
        ]

    def test_mnnvl_partial_commit_still_runs_rank_zero_bounded_phase(self):
        """Once a peer sees MNNVL COMMIT, rank 0 must enter the local phase."""
        import threading
        from contextlib import contextmanager
        from unittest.mock import Mock, patch

        from tensorrt_llm._torch.pyexecutor.py_executor import _SleepWakeupAction
        from tensorrt_llm.executor.base_worker import BaseWorker
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        class FakeComm:
            def __init__(self):
                self.op_id = None
                self.ack_phases = {1: [], 2: []}

            def send(self, payload, dest, tag):
                self.op_id = payload.get("op_id", self.op_id)
                if payload["action"] == _SleepWakeupAction.COMMIT and dest == 2:
                    raise RuntimeError("injected MNNVL commit send failure")
                self.ack_phases[dest].append(payload["action"])

            def iprobe(self, source, tag):
                return True

            def recv(self, source, tag):
                return {
                    "status": "ok",
                    "op_id": self.op_id,
                    "phase": self.ack_phases[source].pop(0),
                    "has_mnnvl_resources": True,
                }

        @contextmanager
        def control_action(**kwargs):
            yield None

        worker = object.__new__(BaseWorker)
        worker._backend = "pytorch"
        worker.rank = 0
        worker.llm_args = SimpleNamespace(
            backend="pytorch",
            parallel_config=SimpleNamespace(world_size=3),
            sleep_config=object(),
        )
        run_mnnvl = Mock()
        worker.engine = SimpleNamespace(
            _sleep_wakeup_lock=threading.Lock(),
            _sleep_wakeup_comm=FakeComm(),
            control_action=control_action,
            _has_mnnvl_checkpoint_resources=lambda tags: True,
            _run_mnnvl_checkpoint_resources=run_mnnvl,
        )
        fail_stop = Mock()
        worker._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag") as release,
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="commit send failure"):
                worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.MODEL_ENGINE_MAIN])

        run_mnnvl.assert_called_once_with(
            _SleepWakeupAction.SLEEP,
            [ExecutorMemoryType.MODEL_ENGINE_MAIN],
        )
        release.assert_called_once_with(ExecutorMemoryType.MODEL_ENGINE_MAIN)
        fail_stop.assert_called_once()

    def test_mnnvl_all_commit_send_errors_are_treated_as_uncertain_delivery(self):
        """A send error cannot prove that no peer received MNNVL COMMIT."""
        import threading
        from contextlib import contextmanager
        from unittest.mock import Mock, patch

        from tensorrt_llm._torch.pyexecutor.py_executor import _SleepWakeupAction
        from tensorrt_llm.executor.base_worker import BaseWorker
        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        class FakeComm:
            def __init__(self):
                self.op_id = None
                self.ack_phases = {1: []}

            def send(self, payload, dest, tag):
                self.op_id = payload.get("op_id", self.op_id)
                if payload["action"] == _SleepWakeupAction.COMMIT:
                    raise RuntimeError("injected MNNVL commit send failure")
                self.ack_phases[dest].append(payload["action"])

            def iprobe(self, source, tag):
                return True

            def recv(self, source, tag):
                return {
                    "status": "ok",
                    "op_id": self.op_id,
                    "phase": self.ack_phases[source].pop(0),
                    "has_mnnvl_resources": True,
                }

        @contextmanager
        def control_action(**kwargs):
            yield None

        worker = object.__new__(BaseWorker)
        worker._backend = "pytorch"
        worker.rank = 0
        worker.llm_args = SimpleNamespace(
            backend="pytorch",
            parallel_config=SimpleNamespace(world_size=2),
            sleep_config=object(),
        )
        run_mnnvl = Mock()
        worker.engine = SimpleNamespace(
            _sleep_wakeup_lock=threading.Lock(),
            _sleep_wakeup_comm=FakeComm(),
            control_action=control_action,
            _has_mnnvl_checkpoint_resources=lambda tags: True,
            _run_mnnvl_checkpoint_resources=run_mnnvl,
        )
        fail_stop = Mock()
        worker._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag") as release,
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="commit send failure"):
                worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.MODEL_ENGINE_MAIN])

        run_mnnvl.assert_called_once_with(
            _SleepWakeupAction.SLEEP,
            [ExecutorMemoryType.MODEL_ENGINE_MAIN],
        )
        release.assert_called_once_with(ExecutorMemoryType.MODEL_ENGINE_MAIN)
        fail_stop.assert_called_once()

    def test_postcommit_non_runtime_error_fail_stops_worker(self):
        """Every ordinary exception after COMMIT must take the fail-stop path."""
        from unittest.mock import Mock, patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        worker, _ = _make_proto_worker(
            [{"status": "ok"}, {"status": "ok"}],
            world_size=2,
        )
        worker.engine._has_mnnvl_checkpoint_resources = lambda tags: True
        worker.engine._run_mnnvl_checkpoint_resources = Mock(
            side_effect=ValueError("invalid checkpoint state")
        )
        fail_stop = Mock()
        worker._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="invalid checkpoint state"):
                worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.MODEL_ENGINE_MAIN])

        fail_stop.assert_called_once()

    def test_divergent_commit_fail_stops_worker(self):
        """A potentially divergent operation poisons the worker before returning."""
        from unittest.mock import patch

        from tensorrt_llm.executor.base_worker import BaseWorker

        worker = object.__new__(BaseWorker)
        worker._fatal_error = None
        worker.engine = SimpleNamespace(
            _fatal_error=None,
            is_shutdown=False,
            fail_sleep_wakeup_transition=MagicMock(),
        )
        error = RuntimeError("divergent sleep state")

        with patch("tensorrt_llm._torch.pyexecutor.hang_detector.propagate_hard_kill") as hard_kill:
            worker._fail_stop_divergent_sleep_wakeup(error)

        assert worker._fatal_error is error
        assert worker.engine._fatal_error is error
        assert worker.engine.is_shutdown
        worker.engine.fail_sleep_wakeup_transition.assert_called_once_with()
        hard_kill.assert_called_once_with()

    def test_precommit_local_failure_remains_recoverable(self):
        """A failure before local mutation aborts peers without fail-stopping."""
        from unittest.mock import Mock, patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        worker, _ = _make_proto_worker(
            [{"status": "ok"}, {"status": "ok"}],
            world_size=2,
        )
        fail_stop = Mock()
        worker._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize", side_effect=RuntimeError("sync failed")),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="sync failed"):
                worker._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        fail_stop.assert_not_called()


class TestListenerUncaughtExceptionSendsErrorAck:
    """An exception that bypasses the narrow except clause must still produce an error ACK.

    The sys.exc_info() check in the finally block detects uncaught exceptions so
    rank-0 receives status=error rather than a false status=ok ACK while the
    listener thread is unwinding.
    """

    def test_uncaught_exception_sends_error_ack(self):
        """MemoryError (not in narrow except) must reach rank-0 as status=error.

        Without the sys.exc_info() guard, the finally block would send status=ok
        because error_msg is still None when the exception bypasses the except
        clause — leaving rank-0 with an inconsistent view of the operation.
        """
        import threading
        from types import SimpleNamespace
        from unittest.mock import patch

        sent_acks = []

        class FakeComm:
            def recv(self, source, tag):
                return {"action": "sleep", "tags": ["kv_cache"]}

            def send(self, payload, dest, tag):
                sent_acks.append(payload)

        # Minimal PyExecutor shell — only the fields the listener loop touches.
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

        executor = object.__new__(PyExecutor)
        executor._sleep_wakeup_comm = FakeComm()
        executor.device_id = 0
        executor.dist = SimpleNamespace(rank=1)
        executor.control_request_barrier = threading.Event()
        executor.control_request_barrier.set()
        executor.control_action_done = threading.Event()
        executor._active_control_id = None

        with (
            patch("torch.cuda.set_device"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.set_thread_local_mpi_comm"),
            patch(
                "tensorrt_llm._torch.virtual_memory.release_with_tag",
                side_effect=MemoryError("simulated OOM outside except list"),
            ),
            patch("torch.cuda.synchronize"),
        ):
            with pytest.raises(MemoryError, match="simulated OOM outside except list"):
                executor._sleep_wakeup_listener_loop()

        assert sent_acks, "finally block must send an ACK even for uncaught exceptions"
        assert sent_acks[0]["status"] == "error", (
            "ACK status must be 'error' when MemoryError bypasses the narrow "
            f"except clause; got {sent_acks[0]!r}"
        )
        assert sent_acks[0]["error"] is not None
        assert "MemoryError" in sent_acks[0]["error"]
        assert not executor.control_request_barrier.is_set()
        assert executor.control_action_done.is_set()


class TestListenerAbortAndShutdown:
    """Listener control messages must unblock cleanly and ACK rank-0."""

    def test_prepare_does_not_execute_vmm_or_release_control(self):
        """Prepare quiesces the peer but leaves VMM and control_action_done untouched."""
        import threading
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _SleepWakeupAction

        sent_acks = []
        op_id = "prepare-only-op"

        class FakeComm:
            def __init__(self):
                self.recv_count = 0

            def recv(self, source, tag):
                self.recv_count += 1
                if self.recv_count > 1:
                    raise StopIteration
                return {
                    "action": _SleepWakeupAction.PREPARE,
                    "target_action": _SleepWakeupAction.SLEEP,
                    "tags": ["kv_cache"],
                    "op_id": op_id,
                }

            def send(self, payload, dest, tag):
                sent_acks.append(payload)

        executor = object.__new__(PyExecutor)
        executor._sleep_wakeup_comm = FakeComm()
        executor.device_id = 0
        executor.dist = SimpleNamespace(rank=1)
        executor.control_request_barrier = threading.Event()
        executor.control_request_barrier.set()
        executor.control_action_done = threading.Event()
        executor._active_control_id = op_id

        with (
            patch("torch.cuda.set_device"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.set_thread_local_mpi_comm"),
            patch("torch.cuda.synchronize"),
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag") as release,
        ):
            try:
                executor._sleep_wakeup_listener_loop()
            except StopIteration:
                pass

        release.assert_not_called()
        assert executor.control_request_barrier.is_set()
        assert not executor.control_action_done.is_set()
        assert sent_acks == [
            {
                "status": "ok",
                "error": None,
                "op_id": op_id,
                "phase": _SleepWakeupAction.PREPARE,
                "has_mnnvl_resources": False,
            }
        ]

    def test_abort_unblocks_control_request_and_sends_success_ack(self):
        """Abort messages release the non-rank executor control barrier."""
        import threading
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _SleepWakeupAction

        sent_acks = []
        op_id = "abort-active-op"

        class FakeComm:
            def __init__(self):
                self.recv_count = 0

            def recv(self, source, tag):
                self.recv_count += 1
                if self.recv_count > 1:
                    raise StopIteration
                return {
                    "action": _SleepWakeupAction.ABORT,
                    "tags": [],
                    "op_id": op_id,
                    "reason": "rank 0 send failed",
                }

            def send(self, payload, dest, tag):
                sent_acks.append(payload)

        executor = object.__new__(PyExecutor)
        executor._sleep_wakeup_comm = FakeComm()
        executor.device_id = 0
        executor.dist = SimpleNamespace(rank=1)
        executor.control_request_barrier = threading.Event()
        executor.control_request_barrier.set()
        executor.control_action_done = threading.Event()
        executor._active_control_id = op_id

        with (
            patch("torch.cuda.set_device"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.set_thread_local_mpi_comm"),
            patch("torch.cuda.synchronize"),
        ):
            try:
                executor._sleep_wakeup_listener_loop()
            except StopIteration:
                pass

        assert executor.control_action_done.is_set()
        assert not executor.control_request_barrier.is_set()
        assert sent_acks
        assert sent_acks[0]["status"] == "ok"
        assert sent_acks[0]["error"] is None
        assert "rank 0 send failed" in sent_acks[0]["reason"]

    def test_abort_before_control_barrier_unblocks_later_control_request(self):
        """An early ABORT is recorded and later consumed by matching control."""
        import threading
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
            CONTROL_REQUEST_ID,
            RequestQueueItem,
        )
        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _SleepWakeupAction

        sent_acks = []
        op_id = "abort-before-barrier-op"

        class FakeComm:
            def __init__(self):
                self.recv_count = 0

            def recv(self, source, tag):
                self.recv_count += 1
                if self.recv_count > 1:
                    raise StopIteration
                return {
                    "action": _SleepWakeupAction.ABORT,
                    "tags": [],
                    "op_id": op_id,
                    "reason": "rank 0 send failed",
                }

            def send(self, payload, dest, tag):
                sent_acks.append(payload)

        executor = object.__new__(PyExecutor)
        executor._sleep_wakeup_comm = FakeComm()
        executor.device_id = 0
        executor.dist = SimpleNamespace(rank=1)
        executor.control_request_barrier = threading.Event()
        executor.control_action_done = threading.Event()
        executor.control_requests = [RequestQueueItem(id=CONTROL_REQUEST_ID, control_id=op_id)]
        executor.active_requests = []
        executor.waiting_queue = []

        with (
            patch("torch.cuda.set_device"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.set_thread_local_mpi_comm"),
            patch("torch.cuda.synchronize"),
        ):
            try:
                executor._sleep_wakeup_listener_loop()
            except StopIteration:
                pass
            assert sent_acks
            assert not executor.control_request_barrier.is_set()

            executor._handle_control_request()

        assert executor.control_requests == []
        assert not executor.control_request_barrier.is_set()
        assert sent_acks
        assert sent_acks[0]["status"] == "ok"
        assert sent_acks[0]["error"] is None
        assert "rank 0 send failed" in sent_acks[0]["reason"]

    def test_shutdown_sends_ack_before_listener_exits(self):
        """Shutdown messages are acknowledged so rank-0 can drain them."""
        from unittest.mock import patch

        from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, _SleepWakeupAction

        sent_acks = []

        class FakeComm:
            def recv(self, source, tag):
                return {"action": _SleepWakeupAction.SHUTDOWN, "tags": []}

            def send(self, payload, dest, tag):
                sent_acks.append(payload)

        executor = object.__new__(PyExecutor)
        executor._sleep_wakeup_comm = FakeComm()
        executor.device_id = 0
        executor.dist = SimpleNamespace(rank=1)

        with (
            patch("torch.cuda.set_device"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.CUASSERT"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.cudart.cudaSetDevice"),
            patch("tensorrt_llm._torch.pyexecutor.py_executor.set_thread_local_mpi_comm"),
        ):
            executor._sleep_wakeup_listener_loop()

        assert sent_acks == [
            {
                "status": "ok",
                "error": None,
                "phase": _SleepWakeupAction.SHUTDOWN,
            }
        ]

    def test_rank0_shutdown_ack_drain_is_bounded(self, monkeypatch):
        """Rank 0 shutdown must not block forever if listener ACKs never arrive."""
        from tensorrt_llm._torch.pyexecutor import py_executor
        from tensorrt_llm._torch.pyexecutor.py_executor import (
            PyExecutor,
            _SleepWakeupAction,
            _SleepWakeupTag,
        )

        sent_messages = []

        class FakeComm:
            def send(self, payload, dest, tag):
                sent_messages.append((payload["action"], dest, tag))

            def iprobe(self, source, tag):
                return False

            def recv(self, source, tag):
                raise AssertionError("shutdown ACK drain must probe before recv")

        executor = object.__new__(PyExecutor)
        executor._sleep_wakeup_comm = FakeComm()
        executor._sleep_wakeup_listener_thread = None
        executor.dist = SimpleNamespace(rank=0, world_size=3)

        monkeypatch.setattr(py_executor, "_SLEEP_WAKEUP_ACK_TIMEOUT_S", 0.0)
        monkeypatch.setattr(py_executor, "_SLEEP_WAKEUP_ACK_POLL_INTERVAL_S", 0.0)

        executor._shutdown_sleep_wakeup_listeners()

        assert sent_messages == [
            (_SleepWakeupAction.SHUTDOWN, 1, _SleepWakeupTag.ACTION),
            (_SleepWakeupAction.SHUTDOWN, 2, _SleepWakeupTag.ACTION),
        ]


class TestMultiRankRank0LocalFailureDrainsAcks:
    """When rank-0's local VMM op raises, all peer ACKs must still be drained.

    Draining before raising keeps the communicator clean for subsequent calls.
    """

    def test_local_failure_drains_all_peer_acks(self):
        """release_with_tag() raises on rank-0; recv() must still be called for every peer.

        If recv() is skipped, stale ACKs would corrupt the next sleep/wakeup call.
        """
        from unittest.mock import patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        # Two peers prepare ok, then abort ok after rank-0 local failure.
        responses = [{"status": "ok"}, {"status": "ok"}]
        w, recv_calls = _make_proto_worker(responses, world_size=3)
        fail_stop = MagicMock()
        w._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch(
                "tensorrt_llm._torch.virtual_memory.release_with_tag",
                side_effect=RuntimeError("rank 0 VMM fault"),
            ),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError, match="rank 0 VMM fault"):
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        # Both peers must have prepare and abort ACKs drained despite the local
        # failure.
        assert len(recv_calls) == 4, (
            f"Expected 4 recv() calls (prepare + abort per peer), got "
            f"{len(recv_calls)}; stale ACKs would corrupt the next "
            "sleep/wakeup call."
        )
        fail_stop.assert_called_once()

    def test_local_failure_plus_peer_error_both_reported(self):
        """When rank-0 fails locally and a peer also errors, both messages must appear.

        Verifies that the error aggregation path collects failures from all sources.
        """
        from unittest.mock import patch

        from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType

        responses = [
            {"status": "ok"},
            {"status": "ok"},
            {"status": "ok"},
            {"status": "error", "error": "rank 2 also failed"},
        ]
        w, _ = _make_proto_worker(responses)
        fail_stop = MagicMock()
        w._fail_stop_divergent_sleep_wakeup = fail_stop

        with (
            patch(
                "tensorrt_llm._torch.virtual_memory.release_with_tag",
                side_effect=RuntimeError("rank 0 VMM fault"),
            ),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                w._multi_rank_sleep_wakeup("sleep", [ExecutorMemoryType.KV_CACHE])

        msg = str(exc_info.value)
        assert "rank 0 VMM fault" in msg
        assert "rank 2 also failed" in msg
        fail_stop.assert_called_once()


class TestSingleRankLockAcquired:
    """sleep() / wakeup() with world_size == 1 must also hold engine._sleep_wakeup_lock.

    The same Event-barrier race exists on the single-rank path.
    """

    @pytest.mark.parametrize("method", ["sleep", "wakeup"])
    def test_single_rank_acquires_lock(self, method):
        """The world_size == 1 branch enters _sleep_wakeup_lock."""
        import threading
        from contextlib import contextmanager
        from unittest.mock import patch

        from tensorrt_llm.executor.base_worker import BaseWorker

        w = object.__new__(BaseWorker)
        w._backend = "pytorch"
        w.llm_args = SimpleNamespace(
            backend="pytorch",
            parallel_config=SimpleNamespace(world_size=1),
            sleep_config=object(),
        )

        lock_entered = []
        real_lock = threading.Lock()

        class SpyLock:
            def __enter__(self_inner):
                lock_entered.append(True)
                return real_lock.__enter__()

            def __exit__(self_inner, *args):
                return real_lock.__exit__(*args)

        @contextmanager
        def _noop_control_action():
            yield None

        w.engine = SimpleNamespace(
            _sleep_wakeup_lock=SpyLock(),
            control_action=_noop_control_action,
            begin_sleep_transition=MagicMock(),
            complete_sleep_transition=MagicMock(),
            abort_sleep_transition=MagicMock(),
            begin_wakeup_transition=MagicMock(),
            complete_wakeup_transition=MagicMock(),
            abort_wakeup_transition=MagicMock(),
            fail_sleep_wakeup_transition=MagicMock(),
        )

        with (
            patch("tensorrt_llm._torch.virtual_memory.release_with_tag"),
            patch("tensorrt_llm._torch.virtual_memory.materialize_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
        ):
            getattr(w, method)(["kv_cache"])

        assert lock_entered, f"{method}() with world_size=1 did not acquire _sleep_wakeup_lock"

    @pytest.mark.parametrize(
        ("method", "mutation"),
        [("sleep", "release_with_tag"), ("wakeup", "materialize_with_tag")],
    )
    def test_post_mutation_failure_permanently_closes_admission(self, method, mutation):
        """A single-rank error after VMM mutation starts is fail-closed."""
        import threading
        from contextlib import contextmanager
        from unittest.mock import patch

        w = _make_worker()

        @contextmanager
        def _noop_control_action():
            yield None

        w.engine._sleep_wakeup_lock = threading.Lock()
        w.engine.control_action = _noop_control_action

        with (
            patch(f"tensorrt_llm._torch.virtual_memory.{mutation}"),
            patch(
                "torch.cuda.synchronize",
                side_effect=[None, RuntimeError("post-mutation sync failed")],
            ),
            patch("gc.collect"),
            patch("torch.cuda.empty_cache"),
            pytest.raises(RuntimeError, match="post-mutation sync failed"),
        ):
            getattr(w, method)(["kv_cache"])

        w.engine.fail_sleep_wakeup_transition.assert_called_once_with()
        getattr(w.engine, f"abort_{method}_transition").assert_called_once_with()
        getattr(w.engine, f"complete_{method}_transition").assert_not_called()


# ---------------------------------------------------------------------------
# GenerationExecutorProxy / GenerationExecutorRpcProxy collective_rpc()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ["ipc", "rpc"])
class TestProxyCollectiveRpcGuards:
    """Guard-path tests for both IPC and RPC proxy collective_rpc() shims."""

    def test_multirank_allowed_for_sleep_wakeup(self, cls):
        """Sleep and wakeup may be called with model_world_size > 1.

        Both are in _MULTI_RANK_ALLOWED_METHODS; the guard must not raise
        and the call must be forwarded to rpc_client.
        """
        from unittest.mock import MagicMock as _MM

        for method_name in ("sleep", "wakeup"):
            mock_call = _MM()
            mock_call.remote.return_value = "ok"
            mock_client = _MM()
            getattr(mock_client, method_name).return_value = mock_call
            p = _make_proxy(cls, model_world_size=2, rpc_client=mock_client)
            result = p.collective_rpc(method_name, args=(["kv_cache"],))
            assert result == ["ok"]

    def test_multirank_raises_for_non_allowlisted_method(self, cls):
        """Non-allowlisted methods still raise NotImplementedError for world_size > 1."""
        p = _make_proxy(cls, model_world_size=2, rpc_client=MagicMock())
        with pytest.raises(NotImplementedError):
            p.collective_rpc("update_weights")

    def test_raises_for_unique_reply_rank(self, cls):
        """Raises NotImplementedError when unique_reply_rank is provided."""
        p = _make_proxy(cls, rpc_client=MagicMock())
        with pytest.raises(NotImplementedError):
            p.collective_rpc("sleep", unique_reply_rank=0)

    def test_raises_for_target_ranks(self, cls):
        """Raises NotImplementedError when target_ranks is provided."""
        p = _make_proxy(cls, rpc_client=MagicMock())
        with pytest.raises(NotImplementedError):
            p.collective_rpc("sleep", target_ranks=[0, 1])

    def test_single_rank_routes_to_rpc_client(self, cls):
        """Blocking call returns [result] and forwards args/kwargs."""
        mock_call = MagicMock()
        mock_call.remote.return_value = "ok"
        mock_client = MagicMock()
        mock_client.my_method.return_value = mock_call

        p = _make_proxy(cls, model_world_size=1, rpc_client=mock_client)
        result = p.collective_rpc("my_method", args=(1,), kwargs={"k": "v"})

        mock_client.my_method.assert_called_once_with(1, k="v")
        assert result == ["ok"]

    def test_single_rank_non_block_returns_future(self, cls):
        """non_block=True returns [Future] without calling .remote()."""
        mock_future = MagicMock()
        mock_call = MagicMock()
        mock_call.remote_future.return_value = mock_future
        mock_client = MagicMock()
        mock_client.my_method.return_value = mock_call

        p = _make_proxy(cls, model_world_size=1, rpc_client=mock_client)
        result = p.collective_rpc("my_method", non_block=True)

        mock_call.remote.assert_not_called()
        assert result == [mock_future]


# IPC proxy additionally validates the rpc_client initialisation guard.
class TestIpcProxyRpcClientGuard:
    """IPC-proxy-specific guard: rpc_client must be initialised."""

    def test_raises_when_rpc_client_is_none(self):
        """Raises RuntimeError when rpc_client has not been set."""
        p = _make_proxy("ipc", rpc_client=None)
        with pytest.raises(RuntimeError, match="RPC client is not initialised"):
            p.collective_rpc("sleep")
