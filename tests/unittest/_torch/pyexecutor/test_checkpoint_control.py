# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native control ordering tests with CUDA and model work kept at the boundary."""

import threading
import time
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mpi4py import MPI
from mpi4py.util import pkl5

from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3ForCausalLM
from tensorrt_llm._torch.pyexecutor.checkpoint import (
    NativeCheckpoint,
    _checkpoint_configuration,
    _CheckpointTag,
    _resident_graph_inventory,
)
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestAdmissionState,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.profiling import ProfileController
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.checkpoint import (
    CheckpointContext,
    CheckpointProfile,
    CheckpointStep,
    RankCheckpointEvidence,
    checkpoint_digest,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs

pytestmark = pytest.mark.cpu_only


def test_pending_pickle_sends_retire_only_after_completion() -> None:
    """Use the pickle request contract and retain an incomplete send across polls."""
    checkpoint = object.__new__(NativeCheckpoint)
    pending = MagicMock(spec=pkl5.Request)
    pending.test.side_effect = [(False, None), (True, None)]
    checkpoint._pending_sends = [pending, pkl5.Request([MPI.REQUEST_NULL])]
    checkpoint._retire_sends()
    assert checkpoint._pending_sends == [pending]
    checkpoint._retire_sends()
    assert checkpoint._pending_sends == []
    assert pending.test.call_count == 2


@pytest.mark.parametrize("completes", [True, False])
def test_shutdown_waits_for_pickle_sends(native_control, completes: bool) -> None:
    """Complete pickle sends or retain their ownership when the close budget expires."""
    pending = MagicMock(spec=pkl5.Request)
    pending.test.return_value = (False, None)
    if completes:
        pending.test.side_effect = [(False, None), (True, None)]
    native_control._pending_sends = [pending]
    if completes:
        native_control.close(timeout=0.1)
        assert native_control._pending_sends == []
        assert pending.test.call_count == 2
    else:
        with pytest.raises(TimeoutError, match="sends are still active"):
            native_control.close(timeout=0.001)
        assert native_control._pending_sends == [pending]
    assert (
        native_control.engine.executor_request_queue.admission_state is RequestAdmissionState.FAILED
    )


@pytest.mark.parametrize("change", ["graph", "metadata", "nested_output"])
def test_unprobed_graph_binding_participates_in_resident_inventory(change) -> None:
    import torch

    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

    # The runner's capture filter selects GPU tensors. Supply just that
    # boundary here; the per-key inventory and pointer comparison are real.
    indices = MagicMock(spec=torch.Tensor)
    indices.is_cuda = True
    indices.data_ptr.return_value = 1000
    owner = SimpleNamespace(indices=indices)
    runner = SimpleNamespace(
        _snapshot_graph_tensor_ptrs=CUDAGraphRunner._snapshot_graph_tensor_ptrs
    )
    runner.graphs = {key: SimpleNamespace(pool=lambda: (1, 0)) for key in (1, 2)}
    runner.graph_metadata = {
        key: {"attn_metadata": owner if key == 2 else None, "spec_metadata": None}
        for key in runner.graphs
    }
    runner.graph_outputs = {key: {"logits": torch.zeros(1)} for key in runner.graphs}
    before = _resident_graph_inventory(runner)
    assert before == _resident_graph_inventory(runner)
    if change == "graph":
        runner.graphs[2] = runner.graphs[1]
    elif change == "metadata":
        indices.data_ptr.return_value = 2000
    else:
        runner.graph_outputs[2]["logits"] = runner.graph_outputs[2]["logits"].clone()
    after = _resident_graph_inventory(runner)
    assert before["1"] == after["1"]
    assert before["2"] != after["2"]


def test_replay_observer_is_scoped_to_private_validation(native_control) -> None:
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner

    class ReplayHooks:
        register_pre_replay_hook = CUDAGraphRunner.register_pre_replay_hook
        unregister_pre_replay_hook = CUDAGraphRunner.unregister_pre_replay_hook

    runner = ReplayHooks()
    other = MagicMock()
    runner._pre_replay_hooks = [other]
    native_control.engine.model_engine = SimpleNamespace(cuda_graph_runner=runner)
    native_control._phase = CheckpointStep.VALIDATION_START
    native_control._start_observing_replays()
    for hook in runner._pre_replay_hooks:
        hook(None, 1, {}, {})
    assert native_control._validation_replays == 1
    native_control._stop_observing_replays()
    native_control._release_checkpoint_control()
    assert runner._pre_replay_hooks == [other]


def test_rank_capacity_is_distinct_from_shared_configuration() -> None:
    """Preserve local GPU budgets without rejecting equivalent shared settings."""
    first = TorchLlmArgs(
        model="/tmp/dummy_model", kv_cache_config=KvCacheConfig(max_gpu_total_bytes=1024)
    )
    second = first.model_copy(deep=True)
    second.kv_cache_config.max_gpu_total_bytes = 2048
    shared_first, local_first = _checkpoint_configuration(first)
    shared_second, local_second = _checkpoint_configuration(second)
    assert shared_first == shared_second
    assert local_first == {"max_gpu_total_bytes": 1024}
    assert local_second == {"max_gpu_total_bytes": 2048}
    assert checkpoint_digest(local_first) != checkpoint_digest(local_second)
    assert first.kv_cache_config.max_gpu_total_bytes == 1024
    second.max_batch_size = first.max_batch_size + 1
    assert _checkpoint_configuration(second)[0] != shared_first


@pytest.mark.parametrize(
    "size,level,passes_thread_check",
    [
        (1, MPI.THREAD_SERIALIZED, True),
        (2, MPI.THREAD_SERIALIZED, False),
        (2, MPI.THREAD_MULTIPLE, True),
    ],
)
def test_thread_support_enters_startup_agreement(monkeypatch, size, level, passes_thread_check):
    """Fail collectively before starting a listener when a peer lacks thread support."""
    rank = size - 1
    engine = SimpleNamespace(_disable_mpi=False, dist=SimpleNamespace(world_size=size, rank=rank))
    comm = MagicMock()
    comm.Dup.return_value = comm
    comm.allgather.side_effect = lambda local: [dict(local, rank=0, error=None), local]
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.checkpoint.mpi_comm", lambda: comm)
    query = MagicMock(return_value=level)
    monkeypatch.setattr(MPI, "Query_thread", query)
    profile = MagicMock(side_effect=RuntimeError("reached profile validation"))
    monkeypatch.setattr(NativeCheckpoint, "_check_profile", profile)
    message = "reached profile validation" if passes_thread_check else "MPI_THREAD_MULTIPLE"
    with pytest.raises(RuntimeError, match=message):
        NativeCheckpoint(engine)
    assert profile.call_count == int(passes_thread_check)
    assert query.call_count == int(size > 1)
    assert comm.allgather.call_count == int(size > 1)


@pytest.fixture
def protected_engine():
    """Use the production queue and control context without starting a GPU loop."""
    engine = object.__new__(PyExecutor)
    engine.dist = SimpleNamespace(rank=0, world_size=1, pp_size=1, cp_size=1)
    engine.executor_request_queue = ExecutorRequestQueue(engine.dist, 4, False, 0)
    engine._control_action_lock = threading.Lock()
    engine.control_request_barrier = threading.Event()
    engine.control_action_done = threading.Event()
    engine.checkpoint = SimpleNamespace(session_id="source")
    engine.executor_request_queue.close_checkpoint_admission("source")
    return engine


def test_checkpoint_memory_status_identifies_failure(protected_engine) -> None:
    """Expose checkpoint failure through the real engine's memory status API."""
    assert protected_engine.get_memory_status() == {
        "state": "running",
        "parked_tags": [],
        "failure_reason": None,
    }
    protected_engine.executor_request_queue.fail_checkpoint()
    assert protected_engine.get_memory_status() == {
        "state": "failed",
        "parked_tags": [],
        "failure_reason": "checkpoint",
    }


def test_competing_control_rejected_before_wait(protected_engine) -> None:
    """An unrelated control action cannot wait behind a persistent capture park."""
    protected_engine._control_action_lock.acquire()
    finished = threading.Event()
    errors = []

    def compete():
        try:
            with protected_engine.control_action():
                errors.append(AssertionError("Competing control entered."))
        except RuntimeError as error:
            errors.append(error)
        finally:
            finished.set()

    thread = threading.Thread(target=compete, daemon=True)
    thread.start()
    try:
        assert finished.wait(1), "Competing control waited for the held scheduler lock."
    finally:
        protected_engine._control_action_lock.release()
        thread.join(timeout=1)
    assert len(errors) == 1 and "Checkpoint owns" in str(errors[0])
    assert protected_engine.executor_request_queue.get_request_queue_size() == 0


def test_timed_control_requires_cancellable_identity(protected_engine) -> None:
    """Reject an anonymous timed sentinel before it can reach the scheduler."""
    with pytest.raises(ValueError, match="control ID"):
        with protected_engine.control_action(
            checkpoint_owner=protected_engine.checkpoint, timeout=1
        ):
            pytest.fail("A timed control needs an identity for cancellation.")
    assert protected_engine.executor_request_queue.get_request_queue_size() == 0


def test_control_timeout_marks_late_sentinel_aborted(protected_engine) -> None:
    """A late scheduler sentinel cannot leave an abandoned capture context parked."""
    with pytest.raises(TimeoutError):
        with protected_engine.control_action(
            checkpoint_owner=protected_engine.checkpoint, control_id="late", timeout=0.001
        ):
            pytest.fail("The scheduler never acknowledged this control.")
    assert protected_engine._pop_sleep_wakeup_abort("late")
    assert protected_engine.get_memory_status()["failure_reason"] == "checkpoint"
    assert not protected_engine.control_action_done.is_set()
    assert protected_engine.executor_request_queue.admission_state is RequestAdmissionState.FAILED
    assert protected_engine._control_action_lock.acquire(blocking=False)
    protected_engine._control_action_lock.release()


@pytest.mark.parametrize("acknowledged", [False, True])
def test_control_timeout_handshake_does_not_leak_completion(
    protected_engine, monkeypatch, acknowledged
) -> None:
    """Cancellation either skips a late sentinel or releases the acknowledged one."""
    engine = protected_engine
    engine._active_control_id = "late"
    if acknowledged:
        assert engine._pop_sleep_wakeup_abort("late", acknowledge=True) is None
    engine._record_sleep_wakeup_abort("late", "timeout", release_active=True)
    assert engine.control_action_done.is_set() == acknowledged
    assert not engine.control_request_barrier.is_set()
    if acknowledged:
        # The parked scheduler consumes the completion before its next iteration.
        engine.control_action_done.clear()
        assert engine._pop_sleep_wakeup_abort("late") is None
    else:
        engine.control_requests = [SimpleNamespace(control_id="late", control_requires_drain=False)]
        engine.active_requests = []
        engine.waiting_queue = []
        engine._retire_inflight_batch_for_control = lambda: None
        monkeypatch.setattr("torch.cuda.synchronize", lambda: None)
        engine._handle_control_request()
    assert not engine.control_action_done.is_set()
    assert not engine.control_request_barrier.is_set()
    assert engine._pop_sleep_wakeup_abort("next", acknowledge=True) is None
    assert engine.control_request_barrier.is_set()
    assert not engine.control_action_done.is_set()


def test_releasing_control_does_not_release_admission(protected_engine) -> None:
    """The independent admission latch outlives the scoped scheduler action."""
    protected_engine.control_request_barrier.set()
    with protected_engine.control_action(
        checkpoint_owner=protected_engine.checkpoint, control_id="ready", timeout=1
    ):
        assert not protected_engine.executor_request_queue.can_enqueue_request()
    assert protected_engine.control_action_done.is_set()
    assert protected_engine.executor_request_queue.checkpoint_admission_closed()


@pytest.mark.parametrize("bound_session", [None, "source", "destination"])
def test_abort_revokes_the_rank_local_session(native_control, bound_session) -> None:
    """ABORT stops ranks before, during and after a partial restore binding."""
    checkpoint = native_control
    queue = checkpoint.engine.executor_request_queue
    checkpoint._phase = CheckpointStep.PREPARE
    checkpoint._baseline = {"resident": "unchanged"}
    checkpoint.profile = CheckpointProfile(
        profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0]
    )
    if bound_session == "destination":
        queue.bind_checkpoint_restore("source", "destination")
    checkpoint.session_id = bound_session
    if bound_session is not None:
        queue.set_checkpoint_validation(bound_session, "probe")
    context = CheckpointContext(
        protocol_version=1,
        operation_id="abort",
        session_id="destination",
        session_kind="restore",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=1,
    )
    evidence = checkpoint._apply_step(
        CheckpointStep.ABORT, context, time.monotonic() + 1, None, None
    )
    assert "progression_stopped" in evidence.checks
    assert evidence.session_id == "destination"
    assert checkpoint._parked is None
    assert checkpoint._failed
    assert not queue.can_enqueue_request()
    assert queue._checkpoint_validation_token is None


@pytest.mark.parametrize(
    "owner",
    [
        "discovery",
        "active_profile",
        "scheduled_profile",
        "completed_profile",
        "rank_reporter",
        "remote_donor",
        "failed_remote_level",
        "shared_weights",
        "mx_weights",
        "pending_profile",
        "resource",
        "cached_nccl",
    ],
)
def test_profile_rejects_present_external_owners(monkeypatch, owner: str) -> None:
    """Absent owners are supported; actual auxiliary owners or profiling are not."""
    for key in (
        "TLLM_NCCL_SYMMETRIC_ZERO_COPY",
        "TRTLLM_PROXY_BATCH_REQUESTS",
        "TRTLLM_GPU_KEEPALIVE",
    ):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr("tensorrt_llm._torch.distributed.ops._NCCL_SYMMETRIC_ZERO_COPY", False)
    manager = object.__new__(KVCacheManagerV2)
    manager._sharing = None
    manager.impl = SimpleNamespace(cache_tier_list=["gpu"])
    engine = SimpleNamespace(
        _rank_reporter=None,
        _disable_mpi=False,
        enable_attention_dp=False,
        dist=SimpleNamespace(world_size=2, pp_size=1, cp_size=1),
        llm_args=SimpleNamespace(
            memory_monitor_config=None,
            num_serve_frontends=1,
            num_postprocess_workers=0,
            enable_resource_governor=False,
            allreduce_strategy="NCCL",
            torch_compile_config=None,
            guided_decoding_backend=None,
            kv_cache_config=SimpleNamespace(kv_events_config=None),
        ),
        model_engine=SimpleNamespace(
            model=object.__new__(Qwen3ForCausalLM),
            model_loader=SimpleNamespace(_gms_backend=None, _checkpoint_loader=None),
        ),
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        draft_model_engine=None,
        drafter=None,
        dwdp_manager=None,
        _openengine_discovery={},
        kv_cache_manager=manager,
        enable_kv_cache_reuse=False,
        resource_manager=SimpleNamespace(
            resource_managers={
                ResourceManagerType.KV_CACHE_MANAGER: manager,
                ResourceManagerType.DRAFT_KV_CACHE_MANAGER: None,
                ResourceManagerType.CROSS_KV_CACHE_MANAGER: None,
            }
        ),
    )
    checkpoint = object.__new__(NativeCheckpoint)
    checkpoint.engine = engine
    engine.get_openengine_discovery = lambda: engine._openengine_discovery
    controller = ProfileController(
        rank=0,
        current_iteration=lambda: 1,
        is_warmup=lambda: False,
        broadcast_start=lambda config: None,
        broadcast_stop=lambda: None,
    )
    controller.profile_start_iters.clear()
    engine._profile_controller = controller
    checkpoint._check_profile()
    if owner == "discovery":
        engine._openengine_discovery = {"publisher": "present"}
    elif owner == "active_profile":
        controller._profile_enabled = True
    elif owner == "scheduled_profile":
        controller.profile_start_iters.add(10)
    elif owner == "completed_profile":
        controller.profile_start_iters.add(0)
        checkpoint._check_profile()
        return
    elif owner == "cached_nccl":
        monkeypatch.setattr("tensorrt_llm._torch.distributed.ops._NCCL_SYMMETRIC_ZERO_COPY", True)
    elif owner == "rank_reporter":
        engine._rank_reporter = object()
    elif owner == "remote_donor":
        manager._remote_donor = object()
    elif owner == "failed_remote_level":
        from tensorrt_llm.runtime.kv_cache_manager_v2 import CacheTier

        manager.impl.cache_tier_list = [CacheTier.GPU_MEM, CacheTier.REMOTE]
        manager.impl.any_remote_level_failed = True
        assert not manager.has_cache_tier_below_gpu
    elif owner == "shared_weights":
        engine.model_engine.model_loader._gms_backend = object()
    elif owner == "mx_weights":
        engine.model_engine.model_loader._checkpoint_loader = SimpleNamespace(
            checkpoint_format="MX"
        )
    elif owner == "pending_profile":
        controller._runtime_profile_pending_start_iter = 0
    else:
        engine.resource_manager.resource_managers[ResourceManagerType.CROSS_KV_CACHE_MANAGER] = (
            object()
        )
    with pytest.raises(
        RuntimeError,
        match=(
            "MX weight"
            if owner == "mx_weights"
            else "external KV|profiling window|resource manager|effective NCCL|rank reports|remote KV|GMS-backed"
        ),
    ):
        checkpoint._check_profile()


def test_rank_rejects_out_of_order_activation(protected_engine) -> None:
    """A valid session alone cannot bypass restore and validation ordering."""
    checkpoint = object.__new__(NativeCheckpoint)
    checkpoint.engine = protected_engine
    checkpoint._phase = CheckpointStep.PREPARE
    ctx = CheckpointContext(
        protocol_version=1,
        operation_id="activate",
        session_id="source",
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=1,
    )
    with pytest.raises(RuntimeError, match="out of order"):
        checkpoint._apply_step(
            CheckpointStep.ACTIVATION_COMMIT, ctx, time.monotonic() + 1, None, None
        )
    assert protected_engine.executor_request_queue.checkpoint_admission_closed()


@pytest.fixture
def native_control(protected_engine):
    """Retain native shutdown ownership with only the scheduler context mocked."""
    checkpoint = object.__new__(NativeCheckpoint)
    checkpoint.engine = protected_engine
    checkpoint.session_id = "source"
    checkpoint._stop = threading.Event()
    checkpoint._command_lock = threading.Lock()
    checkpoint._listener = None
    checkpoint._receive_stopped = False
    checkpoint._parked = MagicMock()
    checkpoint._comm = None
    checkpoint._peers_closed = False
    checkpoint._pending_sends = []
    checkpoint._failed = False
    checkpoint._completed = {}
    checkpoint._phase = None
    return checkpoint


def test_shutdown_retains_resources_while_command_is_active(native_control) -> None:
    """A timed-out shutdown must not unpark a command that still owns the engine."""
    parked = native_control._parked
    native_control._command_lock.acquire()
    try:
        with pytest.raises(TimeoutError, match="command is still active"):
            native_control.close(timeout=0.001)
        parked.__exit__.assert_not_called()
        assert native_control._parked is parked
        assert native_control._stop.is_set()
        assert not native_control.engine.executor_request_queue.can_enqueue_request()
    finally:
        native_control._command_lock.release()
    native_control.close()
    parked.__exit__.assert_called_once_with(None, None, None)
    assert native_control._parked is None
    native_control.close()
    parked.__exit__.assert_called_once()


def test_shutdown_requires_listener_retirement(native_control) -> None:
    """A listener that cannot stop requires host cleanup, not native resource release."""
    native_control._listener = MagicMock(spec=threading.Thread)
    native_control._listener.is_alive.return_value = True
    with pytest.raises(TimeoutError, match="listener is still active"):
        native_control.close(timeout=0.001)


def test_shutdown_timeout_reports_required_supervisor_cleanup(
    protected_engine, monkeypatch
) -> None:
    """Do not release native resources when a checkpoint command still owns them."""
    engine = protected_engine
    engine.worker_started = True
    engine.checkpoint = MagicMock(spec=NativeCheckpoint)
    engine.checkpoint.close.side_effect = TimeoutError("command is still active")
    shutdown = MagicMock()
    monkeypatch.setattr(engine.executor_request_queue, "enqueue_shutdown_request", shutdown)
    log = MagicMock()
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.py_executor.logger.error", log)
    with pytest.raises(TimeoutError, match="command is still active"):
        engine._release_execution_resources()
    shutdown.assert_not_called()
    assert "entire worker job" in log.call_args.args[0]


def test_shutdown_rejects_more_command_work(native_control) -> None:
    """A command cannot continue after shutdown starts, even with unexpired budget."""
    native_control.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        native_control._remaining(time.monotonic() + 10)


def test_revocation_during_restore_rebind(native_control) -> None:
    """Queue ownership can advance before the rank's session field is assigned."""
    queue = native_control.engine.executor_request_queue
    queue.bind_checkpoint_restore("source", "destination")
    queue.set_checkpoint_validation("destination", "probe")
    native_control.fail_closed()
    assert native_control.session_id == "source"
    assert queue.admission_state is RequestAdmissionState.FAILED
    assert queue._checkpoint_validation_token is None


def test_late_cancel_is_not_executable_work(protected_engine: PyExecutor) -> None:
    """A disconnected response can leave cancellation behind a parked sentinel."""
    queue = protected_engine.executor_request_queue
    worker = SimpleNamespace(engine=protected_engine, _client_id_to_request_id={1: 123})
    assert not protected_engine.can_enqueue_requests()
    BaseWorker.abort_request(worker, 1)
    assert queue.get_request_queue_size() == 1
    assert not queue.has_pending_non_cancellation_requests()
    queue.enqueue_control_request(checkpoint_session="source")
    assert queue.has_pending_non_cancellation_requests()
    queue.fail_checkpoint()
    BaseWorker.abort_request(worker, 1)
    assert queue.get_request_queue_size() == 2


def test_standard_loaded_tokenizer_has_stable_profile() -> None:
    """Hash tokenizer behavior, not a runtime object's address or repr."""
    from test_common.serve.fakes import make_fake_tokenizer

    first = TorchLlmArgs(model="/tmp/dummy_model", tokenizer=make_fake_tokenizer())
    second = TorchLlmArgs(model="/tmp/dummy_model", tokenizer=make_fake_tokenizer())
    assert _checkpoint_configuration(first) == _checkpoint_configuration(second)
    second.tokenizer.tokenizer.add_tokens(["new_token"])
    assert _checkpoint_configuration(first) != _checkpoint_configuration(second)


@pytest.mark.parametrize("acknowledged", [True, False])
def test_shutdown_waits_for_peer_release(native_control, acknowledged) -> None:
    """Do not release the leader's resources without peer shutdown evidence."""
    checkpoint = native_control
    checkpoint.engine.dist.world_size = 2
    comm = MagicMock()
    comm.Iprobe.return_value = acknowledged
    comm.isend.return_value.test.return_value = (True, None)
    checkpoint._comm = comm
    parked = checkpoint._parked
    if acknowledged:
        checkpoint.close(timeout=0.1)
        comm.recv.assert_called_once()
        parked.__exit__.assert_called_once()
    else:
        with pytest.raises(TimeoutError, match="peer shutdown"):
            checkpoint.close(timeout=0.001)
        parked.__exit__.assert_not_called()
    comm.isend.assert_called_once()


def test_failed_prepare_cancels_unowned_sentinel(native_control, monkeypatch) -> None:
    """A peer can fail before entering control_action while a sentinel is in flight."""
    checkpoint = native_control
    checkpoint.engine.device_id = 0
    checkpoint._completed = {}
    checkpoint._phase = None
    checkpoint.session_id = None
    checkpoint._parked = None
    checkpoint.profile = CheckpointProfile(
        profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0]
    )
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    checkpoint._check_profile = MagicMock(side_effect=RuntimeError("profiling window"))
    context = CheckpointContext(
        protocol_version=1,
        operation_id="prepare",
        session_id="source",
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=1,
    )
    message = {"step": "prepare", "context": context.model_dump(), "remaining_seconds": 1}
    reply = checkpoint._run_message(message)
    assert reply["error"] == "profiling window"
    assert checkpoint.engine._pop_sleep_wakeup_abort("checkpoint:prepare")
    assert checkpoint._run_message(message) == reply
    checkpoint._check_profile.assert_called_once()


def test_failed_group_does_not_dispatch_late_prepare(native_control) -> None:
    checkpoint = native_control
    checkpoint._failed = True
    checkpoint._comm = MagicMock()
    context = CheckpointContext(
        protocol_version=1,
        operation_id="prepare",
        session_id="source",
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=0,
        time_budget_seconds=1,
    )
    with pytest.raises(RuntimeError, match="group failed"):
        checkpoint.group_step("prepare", context.model_dump(), 1)
    checkpoint._comm.isend.assert_not_called()


@pytest.mark.parametrize("local_close", [False, True])
def test_peer_listener_releases_park_before_shutdown_ack(
    native_control, monkeypatch, local_close: bool
) -> None:
    checkpoint = native_control
    checkpoint.engine.device_id = 0
    checkpoint.engine.dist.rank = 1
    comm = MagicMock()
    comm.Iprobe.return_value = False

    def receive(*, source, tag, status):
        assert source == MPI.ANY_SOURCE and tag == MPI.ANY_TAG
        status.Set_source(0)
        status.Set_tag(_CheckpointTag.CLOSE)
        return 1.0

    comm.recv.side_effect = receive
    checkpoint._comm = comm
    parked = checkpoint._parked
    events = []
    parked.__exit__.side_effect = lambda *args: events.append("release")
    pending = MagicMock(spec=pkl5.Request)
    pending.test.return_value = (True, None)

    def send_ack(*args, **kwargs):
        events.append("ack")
        return pending

    comm.isend.side_effect = send_ack
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.checkpoint.set_thread_local_mpi_comm", lambda comm: None
    )
    checkpoint._listen()
    if local_close:
        # The receive loop has ended, but thread-local cleanup can still be running.
        checkpoint._listener = MagicMock(spec=threading.Thread)
        checkpoint._listener.is_alive.side_effect = [True, False]
        checkpoint.close(timeout=1)
        checkpoint._listener.join.assert_called_once()
    assert events == ["release", "ack"]
    comm.isend.assert_called_once_with(True, dest=0, tag=_CheckpointTag.CLOSE_ACK)
    comm.Iprobe.assert_called_once_with(source=1, tag=_CheckpointTag.CLOSE)
    pending.test.assert_called_once()
    assert checkpoint._pending_sends == []
    assert checkpoint._stop.is_set()
    assert checkpoint.engine.executor_request_queue.admission_state is RequestAdmissionState.FAILED


def test_peer_listener_failure_logs_and_releases_park(native_control, monkeypatch) -> None:
    """A receive error closes admission and retires the park without a logging error."""
    checkpoint = native_control
    checkpoint.engine.device_id = 0
    checkpoint._comm = MagicMock()
    checkpoint._comm.recv.side_effect = RuntimeError("checkpoint receive failed")
    parked = checkpoint._parked
    cleanup_under_lock = []
    parked.__exit__.side_effect = lambda *args: cleanup_under_lock.append(
        checkpoint._command_lock.locked()
    )
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    mpi_context = MagicMock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.checkpoint.set_thread_local_mpi_comm", mpi_context
    )
    log = MagicMock()
    monkeypatch.setattr("tensorrt_llm._torch.pyexecutor.checkpoint.logger.error", log)

    checkpoint._listen()

    assert cleanup_under_lock == [True]
    parked.__exit__.assert_called_once_with(None, None, None)
    assert checkpoint._parked is None
    assert checkpoint._receive_stopped
    assert checkpoint.engine.executor_request_queue.admission_state is RequestAdmissionState.FAILED
    mpi_context.assert_called_with(None)
    log.assert_called_once()
    assert "host cleanup is required" in log.call_args.args[0]
    assert "RuntimeError: checkpoint receive failed" in log.call_args.args[1]


@pytest.mark.parametrize("leader_close", ["absent", "before_local", "after_local"])
def test_local_peer_close_wakes_blocking_listener(
    native_control, monkeypatch, leader_close: str
) -> None:
    """Retire local shutdown and acknowledge a queued leader close in either order."""
    checkpoint = native_control
    checkpoint.engine.dist.rank = 1
    checkpoint.engine.device_id = 0
    inbox = Queue()
    waiting = threading.Event()
    sent = []
    comm = MagicMock()

    def receive(*, source, tag, status=None):
        waiting.set()
        sender, received_tag, message = inbox.get(timeout=2)
        if status is not None:
            status.Set_source(sender)
            status.Set_tag(received_tag)
        else:
            assert (sender, received_tag) == (source, tag)
        return message

    def send(message, *, dest, tag):
        if dest == 1:
            if leader_close == "before_local":
                inbox.put((0, _CheckpointTag.CLOSE, 1.0))
            inbox.put((1, tag, message))
            if leader_close == "after_local":
                inbox.put((0, _CheckpointTag.CLOSE, 1.0))
        else:
            assert dest == 0 and tag == _CheckpointTag.CLOSE_ACK
        pending = MagicMock(spec=pkl5.Request)
        pending.test.return_value = (True, None)
        sent.append(pending)
        return pending

    comm.recv.side_effect = receive
    comm.isend.side_effect = send
    comm.Iprobe.side_effect = lambda **kwargs: not inbox.empty()
    checkpoint._comm = comm
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    monkeypatch.setattr(
        "tensorrt_llm._torch.pyexecutor.checkpoint.set_thread_local_mpi_comm", lambda comm: None
    )
    checkpoint.start()
    try:
        assert waiting.wait(1)
        checkpoint.close(timeout=1)
        assert not checkpoint._listener.is_alive()
        assert checkpoint._parked is None
        assert inbox.empty()
        assert comm.isend.call_args_list[0].kwargs == dict(dest=1, tag=_CheckpointTag.CLOSE)
        assert 0 < comm.isend.call_args_list[0].args[0] <= 1
        assert comm.isend.call_count == (1 if leader_close == "absent" else 2)
        if leader_close != "absent":
            assert comm.isend.call_args_list[1].kwargs == dict(dest=0, tag=_CheckpointTag.CLOSE_ACK)
        assert checkpoint._pending_sends == []
        for pending in sent:
            pending.test.assert_called_once()
    finally:
        if checkpoint._listener.is_alive():
            inbox.put((1, _CheckpointTag.CLOSE, 1.0))
        checkpoint._listener.join(timeout=2)


@pytest.fixture
def activation_group(native_control, monkeypatch):
    """Drive production group coordination and rank transitions over a fake MPI boundary."""
    checkpoint = native_control
    checkpoint.engine.dist.world_size = 2
    checkpoint.engine.device_id = 0
    checkpoint.profile = CheckpointProfile(
        profile_digest="a" * 64, cohort_digest="b" * 64, ranks=[0, 1]
    )
    checkpoint._baseline = {"storage": "unchanged"}
    checkpoint._phase = CheckpointStep.ACTIVATION_PREPARE
    context = CheckpointContext(
        protocol_version=1,
        operation_id="activate",
        session_id="source",
        session_kind="source",
        template_id="template",
        profile_digest="a" * 64,
        cohort_digest="b" * 64,
        expected_state_version=3,
        time_budget_seconds=1,
    )
    reply = dict(
        operation_id=context.operation_id,
        step="activation_commit",
        error=None,
        evidence=RankCheckpointEvidence(
            rank=1,
            session_id=context.session_id,
            profile_digest=context.profile_digest,
            cohort_digest=context.cohort_digest,
            checks=["activation_committed"],
            state_digest="c" * 64,
        ).model_dump(),
    )
    checkpoint._comm = MagicMock()
    checkpoint._comm.Iprobe.return_value = True
    checkpoint._comm.isend.return_value.test.return_value = (True, None)
    checkpoint._comm.recv.return_value = reply
    monkeypatch.setattr("torch.cuda.set_device", lambda device: None)
    return checkpoint, context, reply


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "error",
        "session",
        "profile",
        "cohort",
        "checks",
        "rank",
        "digest",
    ],
)
def test_group_commit_keeps_leader_closed_on_bad_peer_evidence(activation_group, failure) -> None:
    checkpoint, context, reply = activation_group
    parked = checkpoint._parked
    if failure == "missing":
        checkpoint._comm.Iprobe.return_value = False
    elif failure == "error":
        reply["error"] = "peer failed"
    else:
        field, value = {
            "session": ("session_id", "stale"),
            "profile": ("profile_digest", "d" * 64),
            "cohort": ("cohort_digest", "d" * 64),
            "checks": ("checks", ["runtime_valid"]),
            "rank": ("rank", 0),
            "digest": ("state_digest", ""),
        }[failure]
        reply["evidence"][field] = value
    with pytest.raises((RuntimeError, TimeoutError)):
        checkpoint.group_step("activation_commit", context.model_dump(), 0.03)
    parked.__exit__.assert_not_called()
    assert checkpoint.engine.executor_request_queue.admission_state is RequestAdmissionState.FAILED
    assert checkpoint._phase == CheckpointStep.ACTIVATION_PREPARE


def test_group_commit_ignores_stale_ack_and_retries_once(activation_group) -> None:
    checkpoint, context, reply = activation_group
    parked = checkpoint._parked
    responses = iter([{**reply, "operation_id": "old"}, reply, reply])

    def receive(**kwargs):
        response = next(responses)
        if not checkpoint._completed:
            assert checkpoint.engine.executor_request_queue.checkpoint_admission_closed()
            parked.__exit__.assert_not_called()
        return response

    checkpoint._comm.recv.side_effect = receive
    first = checkpoint.group_step("activation_commit", context.model_dump(), 1)
    assert sorted(item["rank"] for item in first) == [0, 1]
    assert not checkpoint.engine.executor_request_queue.checkpoint_admission_closed()
    assert checkpoint.group_step("activation_commit", context.model_dump(), 1) == first
    parked.__exit__.assert_called_once()


def test_group_rejects_conflicting_retry(activation_group) -> None:
    checkpoint, context, _ = activation_group
    checkpoint.group_step("activation_commit", context.model_dump(), 1)
    changed = context.model_copy(update={"template_id": "different"})
    with pytest.raises(RuntimeError, match="Conflicting"):
        checkpoint.group_step("activation_commit", changed.model_dump(), 1)
    assert checkpoint.engine.executor_request_queue.admission_state is RequestAdmissionState.FAILED
