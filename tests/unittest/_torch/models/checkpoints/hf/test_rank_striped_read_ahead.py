# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import sys
import threading
from pathlib import Path
from unittest import mock

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints.hf import rank_striped_read_ahead as read_ahead
from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as weight_loader_module
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@pytest.fixture(autouse=True)
def _isolated_loader_environment(monkeypatch):
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_CACHE", raising=False)
    monkeypatch.delenv("TLLM_OVERRIDE_LAYER_NUM", raising=False)
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", False)
    monkeypatch.setattr(weight_loader_module, "effective_available_host_memory", lambda: 1 << 40)


def _write_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    expected = {
        "model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "model.norm.weight": torch.linspace(0.0, 1.0, 4),
    }
    save_file(expected, str(checkpoint_dir / "model.safetensors"))
    return checkpoint_dir, expected


def _rank_striped_loader(*, partial_model_loading: bool = False) -> HfWeightLoader:
    return HfWeightLoader(
        checkpoint_io_policy="rank_striped_read_ahead",
        partial_model_loading=partial_model_loading,
    )


def test_subclass_without_super_init_uses_native_compatibility_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyLoader(HfWeightLoader):
        def __init__(self) -> None:
            self.custom_state = True

    loader = _LegacyLoader()
    initial_status = loader.last_checkpoint_io_status
    assert loader.checkpoint_io_policy == "native"
    assert initial_status.requested == "native"
    assert initial_status.selected == "native"
    assert initial_status.effective == "none"
    assert loader._partial_model_loading is False

    native_weights = {"native": object()}
    native_load = mock.Mock(return_value=native_weights)
    reader_start = mock.Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", reader_start)

    weights = loader.load_weights("/unused", mapping=Mapping())

    assert weights is native_weights
    native_load.assert_called_once()
    assert loader.last_checkpoint_io_status.effective == "native"
    reader_start.assert_not_called()


def test_sessionless_rank_striped_load_uses_native_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _rank_striped_loader()
    native_weights = {"native": object()}
    native_load = mock.Mock(return_value=native_weights)
    reader_start = mock.Mock(side_effect=AssertionError("must not start"))
    open_weight_session = mock.Mock(side_effect=AssertionError("must not open a session"))
    warning = mock.Mock(side_effect=AssertionError("must not warn"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(loader, "open_weight_session", open_weight_session)
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", reader_start)
    monkeypatch.setattr(weight_loader_module.logger, "warning", warning)
    mapping = Mapping()

    weights = loader.load_weights("/unused", mapping=mapping)

    assert weights is native_weights
    native_load.assert_called_once_with("/unused", mapping, False)
    status = loader.last_checkpoint_io_status
    assert status.requested == "rank_striped_read_ahead"
    assert status.selected == "native"
    assert status.activated is False
    assert status.effective == "native"
    assert "open_weight_session" in status.fallback_reason
    open_weight_session.assert_not_called()
    reader_start.assert_not_called()
    warning.assert_not_called()


def test_checkpoint_io_status_log_escapes_multiline_fallback_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _rank_striped_loader()
    loader._last_checkpoint_io_status.fallback_reason = "first line\r\nsecond line"
    log_info = mock.Mock()
    monkeypatch.setattr(weight_loader_module.logger, "info", log_info)

    loader._log_checkpoint_io_status()

    message = log_info.call_args.args[0]
    assert "\r" not in message
    assert "\n" not in message
    assert r"fallback_reason=first line\r\nsecond line." in message


def test_extent_plan_is_complete_disjoint_and_fair(tmp_path, monkeypatch):
    monkeypatch.setattr(read_ahead, "_CHUNK_SIZE", 4)
    monkeypatch.setattr(read_ahead, "_WORKERS_PER_LOAD_GROUP", 3)
    monkeypatch.setattr(read_ahead, "_WORKERS_PER_RANK", 2)
    files = [(str(tmp_path / "a"), 10), (str(tmp_path / "b"), 7)]
    plans = [read_ahead.build_local_plan(files, rank, 4) for rank in range(4)]

    expected = {
        read_ahead.ReadAheadExtent(str(tmp_path / "a"), 0, 4),
        read_ahead.ReadAheadExtent(str(tmp_path / "a"), 4, 4),
        read_ahead.ReadAheadExtent(str(tmp_path / "a"), 8, 2),
        read_ahead.ReadAheadExtent(str(tmp_path / "b"), 0, 4),
        read_ahead.ReadAheadExtent(str(tmp_path / "b"), 4, 3),
    }
    assigned = [extent for plan in plans for extent in plan.extents]
    assert set(assigned) == expected
    assert len(assigned) == len(expected)
    assert [plan.workers for plan in plans] == [1, 1, 1, 0]

    monkeypatch.setattr(read_ahead, "_WORKERS_PER_LOAD_GROUP", 64)
    monkeypatch.setattr(read_ahead, "_WORKERS_PER_RANK", 16)
    workers = read_ahead.distribute_worker_budget(65)
    assert sum(workers) == 64
    assert max(workers) - min(workers) == 1


@pytest.mark.parametrize(
    ("cgroup", "expected"),
    [
        (
            {
                "/proc/self/cgroup": "0::/jobs/worker\n",
                "/sys/fs/cgroup/jobs/worker/memory.current": "500\n",
                "/sys/fs/cgroup/jobs/worker/memory.max": "1000\n",
                "/sys/fs/cgroup/jobs/worker/memory.high": "700\n",
                "/sys/fs/cgroup/jobs/memory.current": "800\n",
                "/sys/fs/cgroup/jobs/memory.max": "900\n",
            },
            100,
        ),
        (
            {
                "/proc/self/cgroup": "5:memory:/host/path\n",
                "/sys/fs/cgroup/memory/memory.usage_in_bytes": "300\n",
                "/sys/fs/cgroup/memory/memory.limit_in_bytes": "750\n",
            },
            450,
        ),
    ],
)
def test_cgroup_admission_honors_high_ancestors_and_namespaces(monkeypatch, cgroup, expected):
    def read_text(path, *args, **kwargs):
        del args, kwargs
        try:
            return cgroup[str(path)]
        except KeyError as error:
            raise FileNotFoundError(path) from error

    monkeypatch.setattr(Path, "read_text", read_text)
    assert read_ahead.cgroup_available_host_memory() == expected


def test_mapping_mismatch_coordinates_fallback_before_reader_setup(monkeypatch):
    class _Communicator:
        @staticmethod
        def Get_size():
            return 2

        @staticmethod
        def allgather(value):
            return [value, None]

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", _Communicator)

    loader = _rank_striped_loader()
    native_weights = {"native": object()}
    native_load = mock.Mock(return_value=native_weights)
    reader_start = mock.Mock(side_effect=AssertionError("must not start"))
    warning = mock.Mock()
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", reader_start)
    monkeypatch.setattr(weight_loader_module.logger, "warning", warning)

    with loader.open_weight_session("/unused", mapping=Mapping(world_size=1)) as weights:
        assert weights is native_weights

    native_load.assert_called_once()
    assert native_load.call_args.kwargs["_allow_prefetch"] is False
    status = loader.last_checkpoint_io_status
    assert status.selected == "native"
    assert status.effective == "native"
    assert "mapping.world_size" in status.fallback_reason
    reader_start.assert_not_called()
    warning.assert_called_once()


def test_communicator_failure_does_not_fall_back_locally(monkeypatch):
    class _Communicator:
        @staticmethod
        def Get_size():
            raise RuntimeError("MPI communicator failure")

    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", _Communicator)

    loader = _rank_striped_loader()
    native_load = mock.Mock()
    monkeypatch.setattr(loader, "_load_weights_native", native_load)

    with pytest.raises(RuntimeError, match="MPI communicator failure"):
        with loader.open_weight_session("/unused", mapping=Mapping()):
            pass

    native_load.assert_not_called()


@pytest.mark.parametrize("enable_multi_device", [True, False])
def test_mpi_disabled_distributed_fallback_preserves_native_prefetch(
    monkeypatch: pytest.MonkeyPatch, enable_multi_device: bool
) -> None:
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", enable_multi_device)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: True)
    monkeypatch.setattr(
        weight_loader_module,
        "mpi_comm",
        mock.Mock(side_effect=AssertionError("must not communicate")),
    )

    loader = HfWeightLoader(
        checkpoint_io_policy="rank_striped_read_ahead",
        requested_checkpoint_io_policy="auto",
    )
    native_weights = {"native": object()}
    native_load = mock.Mock(return_value=native_weights)
    reader_start = mock.Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", reader_start)

    with loader.open_weight_session("/unused", mapping=Mapping(world_size=2, tp_size=2)) as weights:
        assert weights is native_weights

    native_load.assert_called_once()
    assert native_load.call_args.kwargs["_allow_prefetch"] is True
    assert native_load.call_args.kwargs["_local_communicator"] is None
    status = loader.last_checkpoint_io_status
    assert status.requested == "auto"
    assert status.selected == "native"
    assert status.effective == "native"
    assert "active MPI communicator" in status.fallback_reason
    reader_start.assert_not_called()


def test_collective_native_fallback_failure_escapes_without_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_communicator = mock.Mock()
    active_communicator.Get_size.return_value = 2
    active_communicator.allgather.side_effect = lambda value: [value, value]
    node_communicator = mock.Mock()
    node_communicator.Get_size.return_value = 2

    loader = _rank_striped_loader()
    native_load = mock.Mock(side_effect=OSError("rank-local prefetch failure"))
    close_node_communicator = mock.Mock()
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(weight_loader_module, "close_node_communicator", close_node_communicator)

    with pytest.raises(OSError, match="rank-local prefetch failure"):
        loader._fallback_to_native(
            "/unused",
            Mapping(world_size=2, tp_size=2),
            False,
            "runtime fallback",
            active_communicator,
            node_communicator,
            allow_native_prefetch=True,
        )

    native_load.assert_called_once()
    assert native_load.call_args.kwargs["_local_communicator"] is node_communicator
    assert native_load.call_args.kwargs["_allow_prefetch"] is True
    # The only active consensus happens before native loading, while every rank
    # is still known to be in the same fallback phase.
    active_communicator.allgather.assert_called_once_with(None)
    close_node_communicator.assert_not_called()


@pytest.mark.parametrize("requested_policy", ["auto", "rank_striped_read_ahead"])
@pytest.mark.parametrize("partial,available", [(True, 1 << 40), (False, 0)])
def test_ineligible_request_uses_exact_native_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_policy: str,
    partial: bool,
    available: int,
) -> None:
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = HfWeightLoader(
        checkpoint_io_policy="rank_striped_read_ahead",
        requested_checkpoint_io_policy=requested_policy,
        partial_model_loading=partial,
    )
    native_weights = {"native": object()}
    native_load = mock.Mock(return_value=native_weights)
    reader_start = mock.Mock(side_effect=AssertionError("must not start"))
    warning = mock.Mock()
    info = mock.Mock()
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", reader_start)
    monkeypatch.setattr(weight_loader_module.logger, "warning", warning)
    monkeypatch.setattr(weight_loader_module.logger, "info", info)
    monkeypatch.setattr(weight_loader_module, "effective_available_host_memory", lambda: available)

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as weights:
        assert weights is native_weights

    native_load.assert_called_once()
    status = loader.last_checkpoint_io_status
    assert status.selected == "native"
    assert not status.activated
    assert status.effective == "native"
    assert status.fallback_reason
    reader_start.assert_not_called()
    if requested_policy == "rank_striped_read_ahead":
        warning.assert_called_once()
        assert "falling back before model materialization" in warning.call_args.args[0]
    else:
        warning.assert_not_called()
        assert any(
            "falling back before model materialization" in call.args[0]
            for call in info.call_args_list
        )


def test_disabled_weight_cache_does_not_force_fallback(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "1")
    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES", "0")

    loader = _rank_striped_loader()
    loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert loader.last_checkpoint_io_status.activated


def test_reader_start_failure_cleans_up_and_falls_back(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    native_weights = {"native": object()}
    native_load = mock.Mock(return_value=native_weights)
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    close_file_descriptors = []
    original_close = read_ahead.RankStripedReadAheadSession._close_file_descriptors

    def record_close(session):
        close_file_descriptors.append(True)
        return original_close(session)

    monkeypatch.setattr(
        read_ahead.RankStripedReadAheadSession,
        "_close_file_descriptors",
        record_close,
    )
    monkeypatch.setattr(
        read_ahead.threading.Thread,
        "start",
        mock.Mock(side_effect=RuntimeError("start failed")),
    )

    weights = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert weights is native_weights
    assert close_file_descriptors
    native_load.assert_called_once()
    status = loader.last_checkpoint_io_status
    assert status.selected == "rank_striped_read_ahead"
    assert not status.activated
    assert status.effective == "native"
    assert "start failed" in status.fallback_reason


def test_session_overlaps_materialization_and_cancels_tail(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    started = threading.Event()
    stopped = threading.Event()

    def wait_for_cancel(session, _extent):
        started.set()
        assert session._cancel.wait(timeout=5)
        stopped.set()
        return 0

    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "_read_extent", wait_for_cancel)

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as weights:
        assert started.wait(timeout=5)
        assert not stopped.is_set()
        assert "model.norm.weight" in weights

    assert stopped.is_set()
    assert loader.last_checkpoint_io_status.effective == "rank_striped_read_ahead"


def test_advisory_read_failure_keeps_materialized_weights(tmp_path, monkeypatch):
    checkpoint_dir, expected = _write_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    native_load = mock.Mock(side_effect=AssertionError("must not reload"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(
        read_ahead.RankStripedReadAheadSession,
        "_read_extent",
        mock.Mock(side_effect=OSError("injected read failure")),
    )

    weights = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert torch.equal(weights["model.norm.weight"], expected["model.norm.weight"])
    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.activated
    assert status.effective == "native"
    assert "injected read failure" in status.fallback_reason


def test_mapping_and_materialization_failures_never_retry(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    native_load = mock.Mock(side_effect=AssertionError("must not reload"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(
        loader, "_load_weights_in_parallel", mock.Mock(side_effect=RuntimeError("mapping failed"))
    )

    with pytest.raises(RuntimeError, match="mapping failed"):
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())
    native_load.assert_not_called()

    loader = _rank_striped_loader()
    native_load = mock.Mock(side_effect=AssertionError("must not reload"))
    error_log = mock.Mock()
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(weight_loader_module.logger, "error", error_log)
    with pytest.raises(RuntimeError, match="materialization failed"):
        with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()):
            raise RuntimeError("materialization failed")
    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.effective == "none"
    assert status.fallback_reason == (
        "model materialization failed: RuntimeError: materialization failed"
    )
    assert "model materialization failed" in error_log.call_args.args[0]
    assert "cleanup failure" not in error_log.call_args.args[0]


def test_native_and_rank_striped_outputs_match(tmp_path, monkeypatch):
    # CPU-only CI has no CUDA device from which local_mpi_rank can derive a
    # rank. Model the single-rank environment explicitly; this test validates
    # loader output parity, not MPI rank discovery.
    monkeypatch.setattr(weight_loader_module, "local_mpi_rank", lambda: 0)
    monkeypatch.setattr(weight_loader_module, "local_mpi_size", lambda: 1)
    monkeypatch.setattr(weight_loader_module, "local_mpi_barrier", lambda: None)

    checkpoint_dir, expected = _write_checkpoint(tmp_path)
    native_loader = HfWeightLoader()
    native = native_loader.load_weights(str(checkpoint_dir), mapping=Mapping())
    striped = _rank_striped_loader().load_weights(str(checkpoint_dir), mapping=Mapping())

    assert native_loader.last_checkpoint_io_status.effective == "native"
    assert set(native) == set(striped) == set(expected)
    for name, expected_tensor in expected.items():
        assert torch.equal(native[name], expected_tensor)
        assert torch.equal(striped[name], expected_tensor)


def _run_real_mpi_scenarios(checkpoint_dir: str) -> dict:
    from tensorrt_llm._torch.models.checkpoints.hf import rank_striped_read_ahead, weight_loader
    from tensorrt_llm._utils import mpi_comm, set_thread_local_mpi_comm

    communicator = mpi_comm()
    rank = communicator.Get_rank()
    world_size = communicator.Get_size()
    mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
    weight_loader.effective_available_host_memory = lambda: 1 << 40
    rank_striped_read_ahead._CHUNK_SIZE = 8

    success_loader = _rank_striped_loader()
    success_weights = success_loader.load_weights(checkpoint_dir, mapping=mapping)
    success = success_loader.last_checkpoint_io_status.effective
    communicator.Barrier()

    fallback_failure_loader = _rank_striped_loader()
    if rank == world_size - 1:
        original_native_load = fallback_failure_loader._load_weights_native

        def fail_after_native_load(*args, **kwargs):
            original_native_load(*args, **kwargs)
            raise RuntimeError("rank-local native fallback failure")

        fallback_failure_loader._load_weights_native = fail_after_native_load
    fallback_failure_error = None
    try:
        fallback_failure_loader.load_weights(
            checkpoint_dir, mapping=Mapping(world_size=1, rank=0, tp_size=1)
        )
    except BaseException as error:
        fallback_failure_error = str(error)
    communicator.Barrier()

    subgroup = communicator.Split(color=rank, key=0)
    set_thread_local_mpi_comm(subgroup)
    subgroup_error = None
    try:
        subgroup_loader = _rank_striped_loader(partial_model_loading=rank == 0)
        subgroup_weights = subgroup_loader.load_weights(
            checkpoint_dir, mapping=Mapping(world_size=1, rank=0, tp_size=1)
        )
        subgroup_effective = subgroup_loader.last_checkpoint_io_status.effective
    except BaseException as error:
        subgroup_error = f"{type(error).__name__}: {error}"
    finally:
        set_thread_local_mpi_comm(None)
        subgroup.Free()
    subgroup_errors = communicator.allgather(subgroup_error)
    if any(error is not None for error in subgroup_errors):
        raise RuntimeError(f"Subgroup fallback scenario failed: {subgroup_errors}")

    body_loader = _rank_striped_loader()
    body_error = None
    try:
        with body_loader.open_weight_session(checkpoint_dir, mapping=mapping) as weights:
            assert weights
            if rank == world_size - 1:
                raise RuntimeError("rank-local body failure")
    except RuntimeError as error:
        body_error = str(error)
    communicator.Barrier()

    original_read_extent = rank_striped_read_ahead.RankStripedReadAheadSession._read_extent
    original_run = rank_striped_read_ahead.RankStripedReadAheadSession._run
    read_error_recorded = threading.Event()
    if rank == world_size - 1:

        def fail_read(_session, _extent):
            raise OSError("rank-local read failure")

        def run_and_record_error(session):
            original_run(session)
            if session._read_error is not None:
                read_error_recorded.set()

        rank_striped_read_ahead.RankStripedReadAheadSession._read_extent = fail_read
        rank_striped_read_ahead.RankStripedReadAheadSession._run = run_and_record_error
    read_loader = _rank_striped_loader()
    try:
        with read_loader.open_weight_session(checkpoint_dir, mapping=mapping) as read_weights:
            local_error_recorded = rank != world_size - 1 or read_error_recorded.wait(timeout=5)
            error_recorded = communicator.allgather(local_error_recorded)
            if not all(error_recorded):
                raise RuntimeError("Timed out waiting for the injected read failure.")
        read_status = read_loader.last_checkpoint_io_status
    finally:
        rank_striped_read_ahead.RankStripedReadAheadSession._read_extent = original_read_extent
        rank_striped_read_ahead.RankStripedReadAheadSession._run = original_run
    communicator.Barrier()

    return {
        "rank": rank,
        "success": success,
        "sum": float(success_weights["model.norm.weight"].sum()),
        "fallback_failure_error": fallback_failure_error,
        "subgroup_effective": subgroup_effective,
        "subgroup_sum": float(subgroup_weights["model.norm.weight"].sum()),
        "body_error": body_error,
        "read_effective": read_status.effective,
        "read_reason": read_status.fallback_reason,
        "read_sum": float(read_weights["model.norm.weight"].sum()),
    }


@pytest.mark.skipif(
    not weight_loader_module.ENABLE_MULTI_DEVICE or weight_loader_module.mpi_disabled(),
    reason="requires an MPI-enabled TensorRT-LLM build",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_real_mpi_success_and_rank_local_failures(tmp_path, mpi_pool_executor):
    checkpoint_dir, expected = _write_checkpoint(tmp_path)
    results = list(
        mpi_pool_executor.map(
            _run_real_mpi_scenarios,
            [str(checkpoint_dir)] * mpi_pool_executor.num_workers,
        )
    )

    expected_sum = float(expected["model.norm.weight"].sum())
    assert {result["rank"] for result in results} == {0, 1}
    assert all(result["success"] == "rank_striped_read_ahead" for result in results)
    assert all(result["sum"] == expected_sum for result in results)
    assert all(
        "rank-local native fallback failure" in result["fallback_failure_error"]
        for result in results
    )
    assert {result["subgroup_effective"] for result in results} == {
        "native",
        "rank_striped_read_ahead",
    }
    assert all(result["subgroup_sum"] == expected_sum for result in results)
    assert all(result["body_error"] is not None for result in results)
    assert all(result["read_effective"] == "native" for result in results)
    assert all("rank-local read failure" in result["read_reason"] for result in results)
    assert all(result["read_sum"] == expected_sum for result in results)
