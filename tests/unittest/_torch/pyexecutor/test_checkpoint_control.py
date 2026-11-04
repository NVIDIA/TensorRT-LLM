# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native control ordering tests with CUDA and model work kept at the boundary."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mpi4py import MPI
from mpi4py.util import pkl5

from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3ForCausalLM
from tensorrt_llm._torch.pyexecutor.checkpoint import NativeCheckpoint, _checkpoint_configuration
from tensorrt_llm._torch.pyexecutor.executor_request_queue import ExecutorRequestQueue
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.executor.checkpoint import CheckpointContext, CheckpointStep, checkpoint_digest
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


def test_competing_control_rejected_before_wait(protected_engine) -> None:
    """An unrelated control action cannot wait behind a persistent capture park."""
    with pytest.raises(RuntimeError, match="Checkpoint owns"):
        with protected_engine.control_action():
            pytest.fail("Competing control entered the protected boundary.")
    assert protected_engine.executor_request_queue.get_request_queue_size() == 0


def test_control_timeout_marks_late_sentinel_aborted(protected_engine) -> None:
    """A late scheduler sentinel cannot leave an abandoned capture context parked."""
    with pytest.raises(TimeoutError):
        with protected_engine.control_action(
            checkpoint_owner=protected_engine.checkpoint, control_id="late", timeout=0.001
        ):
            pytest.fail("The scheduler never acknowledged this control.")
    assert protected_engine._pop_sleep_wakeup_abort("late")
    assert not protected_engine.executor_request_queue.can_enqueue_request()
    assert protected_engine._control_action_lock.acquire(blocking=False)
    protected_engine._control_action_lock.release()


def test_releasing_control_does_not_release_admission(protected_engine) -> None:
    """The independent admission latch outlives the scoped scheduler action."""
    protected_engine.control_request_barrier.set()
    with protected_engine.control_action(
        checkpoint_owner=protected_engine.checkpoint, control_id="ready", timeout=1
    ):
        assert not protected_engine.executor_request_queue.can_enqueue_request()
    assert protected_engine.control_action_done.is_set()
    assert protected_engine.executor_request_queue.checkpoint_admission_closed()


@pytest.mark.parametrize("owner", ["discovery", "active_profile", "scheduled_profile", "resource"])
def test_profile_rejects_present_external_owners(monkeypatch, owner: str) -> None:
    """Absent owners are supported; actual auxiliary owners or profiling are not."""
    for key in (
        "TLLM_NCCL_SYMMETRIC_ZERO_COPY",
        "TRTLLM_PROXY_BATCH_REQUESTS",
        "TRTLLM_GPU_KEEPALIVE",
    ):
        monkeypatch.setenv(key, "0")
    manager = object.__new__(KVCacheManagerV2)
    manager._sharing = None
    manager.impl = SimpleNamespace(cache_tier_list=["gpu"])
    engine = SimpleNamespace(
        _profile_state_lock=threading.Lock(),
        _profile_enabled=False,
        _runtime_profile_pending_start_iter=None,
        profile_start_iters=set(),
        _disable_mpi=False,
        enable_attention_dp=False,
        dist=SimpleNamespace(world_size=2, pp_size=1, cp_size=1),
        llm_args=SimpleNamespace(
            num_serve_frontends=1,
            num_postprocess_workers=0,
            enable_resource_governor=False,
            allreduce_strategy="NCCL",
            torch_compile_config=None,
            guided_decoding_backend=None,
            kv_cache_config=SimpleNamespace(kv_events_config=None),
        ),
        model_engine=SimpleNamespace(model=object.__new__(Qwen3ForCausalLM)),
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
    checkpoint._check_profile()
    if owner == "discovery":
        engine._openengine_discovery = {"publisher": "present"}
    elif owner == "active_profile":
        engine._profile_enabled = True
    elif owner == "scheduled_profile":
        engine.profile_start_iters.add(10)
    else:
        engine.resource_manager.resource_managers[ResourceManagerType.CROSS_KV_CACHE_MANAGER] = (
            object()
        )
    with pytest.raises(RuntimeError, match="external KV|profiling window|resource manager"):
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
    checkpoint._parked = MagicMock()
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


def test_shutdown_rejects_more_command_work(native_control) -> None:
    """A command cannot continue after shutdown starts, even with unexpired budget."""
    native_control.close()
    with pytest.raises(RuntimeError, match="shutting down"):
        native_control._remaining(time.monotonic() + 10)
