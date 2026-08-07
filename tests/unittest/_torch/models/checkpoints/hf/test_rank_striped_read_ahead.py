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

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as weight_loader_module
from tensorrt_llm._torch.pyexecutor.model_loader import _construct_checkpoint_loader
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _clear_checkpoint_loader_environment(monkeypatch):
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_CACHE", raising=False)
    monkeypatch.delenv("TLLM_OVERRIDE_LAYER_NUM", raising=False)


def _write_tiny_checkpoint(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    expected = {
        "model.embed_tokens.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
        "model.norm.weight": torch.linspace(0.0, 1.0, 4),
    }
    save_file(expected, str(checkpoint_dir / "model.safetensors"))
    return checkpoint_dir, expected


def _rank_striped_loader(**kwargs) -> HfWeightLoader:
    return HfWeightLoader(
        checkpoint_io_policy="rank_striped_read_ahead",
        host_memory_headroom_bytes=0,
        host_memory_headroom_fraction=0.0,
        prefetch_chunk_size=8,
        **kwargs,
    )


def _run_rank_striped_mpi_smoke(checkpoint_dir: str) -> dict:
    """Exercise real active-communicator collectives in an MPI worker."""
    from tensorrt_llm._utils import mpi_comm

    communicator = mpi_comm()
    rank = communicator.Get_rank()
    world_size = communicator.Get_size()

    divergent_policy = "native" if rank == 0 else "rank_striped_read_ahead"
    divergent_loader = HfWeightLoader(checkpoint_io_policy=divergent_policy)
    try:
        divergent_loader.coordinate_checkpoint_io_request(SimpleNamespace(world_size=world_size))
    except RuntimeError as error:
        policy_error = str(error)
    else:
        raise AssertionError("rank-divergent policies were not rejected")
    communicator.Barrier()

    divergent_mapping = world_size + (rank == world_size - 1)
    mapping_loader = HfWeightLoader()
    try:
        mapping_loader.coordinate_checkpoint_io_request(
            SimpleNamespace(world_size=divergent_mapping)
        )
    except RuntimeError as error:
        mapping_error = str(error)
    else:
        raise AssertionError("rank-divergent mappings were not rejected")
    communicator.Barrier()

    mapping = Mapping(world_size=world_size, rank=rank, tp_size=world_size)
    loader = _rank_striped_loader(prefetch_workers_per_node=world_size)
    observed_chunks = []
    prefetch_one_chunk = loader._prefetch_one_chunk

    def record_prefetch(file_name, offset, length, cancel_event=None):
        observed_chunks.append((file_name, offset, length))
        prefetch_one_chunk(file_name, offset, length, cancel_event=cancel_event)

    loader._prefetch_one_chunk = record_prefetch
    loader.coordinate_checkpoint_io_request(mapping)
    weights = loader.load_weights(checkpoint_dir, mapping=mapping)
    status = loader.last_checkpoint_io_status
    communicator.Barrier()
    return {
        "rank": rank,
        "policy_error": policy_error,
        "mapping_error": mapping_error,
        "chunks": sorted(observed_chunks),
        "effective": status.effective,
        "assigned_bytes": status.assigned_bytes,
        "completed_bytes": status.completed_bytes,
        "embed_shape": tuple(weights["model.embed_tokens.weight"].shape),
        "embed_sum": float(weights["model.embed_tokens.weight"].sum()),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_io_policy": "unknown"},
        {"prefetch_chunk_size": 0},
        {"prefetch_workers_per_node": 0},
        {"prefetch_workers_per_rank": 0},
        {"host_memory_headroom_bytes": -1},
        {"host_memory_headroom_fraction": -0.1},
        {"host_memory_headroom_fraction": 1.0},
    ],
)
def test_constructor_rejects_invalid_io_settings(kwargs):
    with pytest.raises(ValueError):
        HfWeightLoader(**kwargs)


def test_construct_checkpoint_loader_propagates_rank_striped_policy():
    checkpoint_loader = _construct_checkpoint_loader(
        "pytorch",
        checkpoint_loader=None,
        checkpoint_format="HF",
        checkpoint_io_policy="rank_striped_read_ahead",
    )

    assert isinstance(checkpoint_loader.weight_loader, HfWeightLoader)
    assert checkpoint_loader.weight_loader.checkpoint_io_policy == "rank_striped_read_ahead"


def test_rank_divergent_policy_is_rejected_before_loading(monkeypatch):
    class _FakeWorldCommunicator:
        @staticmethod
        def Get_size():
            return 2

        @staticmethod
        def allgather(_request):
            return [
                ("native", 2, 2),
                ("rank_striped_read_ahead", 2, 2),
            ]

    loader = HfWeightLoader()
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", _FakeWorldCommunicator)

    with pytest.raises(RuntimeError, match="checkpoint_io_policy.*must match"):
        loader.coordinate_checkpoint_io_request(SimpleNamespace(world_size=2))


def test_rank_divergent_mapping_is_rejected_coherently(monkeypatch):
    class _FakeWorldCommunicator:
        @staticmethod
        def Get_size():
            return 2

        @staticmethod
        def allgather(_request):
            return [("native", 2, 2), ("native", 3, 2)]

    loader = HfWeightLoader()
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", _FakeWorldCommunicator)

    with pytest.raises(RuntimeError, match="every mapping.world_size"):
        loader.coordinate_checkpoint_io_request(SimpleNamespace(world_size=2))


def test_standalone_native_loader_does_not_add_global_policy_collective(monkeypatch):
    loader = HfWeightLoader()
    world_communicator = mock.Mock()
    monkeypatch.setattr(weight_loader_module, "ENABLE_MULTI_DEVICE", True)
    monkeypatch.setattr(weight_loader_module, "mpi_disabled", lambda: False)
    monkeypatch.setattr(weight_loader_module, "mpi_comm", lambda: world_communicator)
    native_weights = {"native": object()}

    with mock.patch.object(loader, "_load_weights_native", return_value=native_weights):
        weights = loader.load_weights("/checkpoint", mapping=Mapping())

    assert weights is native_weights
    world_communicator.allgather.assert_not_called()


def test_worker_budget_is_exact_and_evenly_distributed():
    assert HfWeightLoader._distribute_worker_budget(8, 64, 16) == (8,) * 8
    assert HfWeightLoader._distribute_worker_budget(5, 7, 16) == (2, 2, 1, 1, 1)

    workers = HfWeightLoader._distribute_worker_budget(65, 64, 16)
    assert sum(workers) == 64
    assert max(workers) - min(workers) == 1
    assert workers[:64] == (1,) * 64
    assert workers[64] == 0


@pytest.mark.parametrize(
    "args",
    [
        (0, 64, 16),
        (8, 0, 16),
        (8, 64, 0),
    ],
)
def test_worker_budget_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        HfWeightLoader._distribute_worker_budget(*args)


def test_rank_plans_cover_each_chunk_exactly_once(tmp_path):
    first_file = tmp_path / "a.safetensors"
    second_file = tmp_path / "b.safetensors"
    first_file.write_bytes(b"a" * 10)
    second_file.write_bytes(b"b" * 7)
    files = [str(second_file), str(first_file)]

    loader = HfWeightLoader(
        checkpoint_io_policy="rank_striped_read_ahead",
        prefetch_chunk_size=4,
        prefetch_workers_per_node=3,
        prefetch_workers_per_rank=2,
    )
    plans = [loader._local_prefetch_plan(files, rank, 4) for rank in range(4)]

    expected_chunks = {
        (str(first_file), 0, 4),
        (str(first_file), 4, 4),
        (str(first_file), 8, 2),
        (str(second_file), 0, 4),
        (str(second_file), 4, 3),
    }
    assigned_chunks = [chunk for chunks, _ in plans for chunk in chunks]

    assert set(assigned_chunks) == expected_chunks
    assert len(assigned_chunks) == len(expected_chunks)
    assert [workers for _, workers in plans] == [1, 1, 1, 0]


def test_cgroup_v2_available_memory_uses_relative_process_path(monkeypatch):
    contents = {
        "/proc/self/cgroup": "0::/job.slice/worker.scope\n",
        "/sys/fs/cgroup/job.slice/worker.scope/memory.max": "1000\n",
        "/sys/fs/cgroup/job.slice/worker.scope/memory.current": "375\n",
    }

    def fake_read_text(path: Path, *args, **kwargs) -> str:
        del args, kwargs
        try:
            return contents[str(path)]
        except KeyError as error:
            raise FileNotFoundError(path) from error

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert HfWeightLoader._get_cgroup_available_host_memory() == 625


def test_effective_available_memory_is_capped_by_cgroup(monkeypatch):
    monkeypatch.setattr(
        weight_loader_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=1000),
    )
    monkeypatch.setattr(
        HfWeightLoader,
        "_get_cgroup_available_host_memory",
        staticmethod(lambda: 600),
    )
    assert HfWeightLoader._get_effective_available_host_memory() == 600

    monkeypatch.setattr(
        HfWeightLoader,
        "_get_cgroup_available_host_memory",
        staticmethod(lambda: None),
    )
    assert HfWeightLoader._get_effective_available_host_memory() == 1000


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("lazy", "model-specific lazy SafeTensors"),
        ("cache", "raw HF weight cache"),
        ("partial", "partial checkpoint loading"),
        ("bin", ".bin checkpoints"),
    ],
)
def test_ineligible_checkpoint_paths_select_native_before_start(
    tmp_path, monkeypatch, setup, reason
):
    checkpoint_dir = tmp_path / setup
    checkpoint_dir.mkdir()
    if setup == "bin":
        (checkpoint_dir / "model.bin").touch()
    else:
        (checkpoint_dir / "model.safetensors").touch()
    if setup == "lazy":
        (checkpoint_dir / "config.json").write_text('{"model_type": "kimi_k3"}')
    if setup == "cache":
        monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "1")
    if setup == "partial":
        monkeypatch.setenv("TLLM_OVERRIDE_LAYER_NUM", "1")

    loader = _rank_striped_loader()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)
    native_weights = {"native": object()}
    with (
        mock.patch.object(
            loader, "_load_weights_native", return_value=native_weights
        ) as native_load,
        mock.patch.object(loader, "_prefetch_chunks") as prefetch_chunks,
    ):
        weights = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert weights is native_weights
    native_load.assert_called_once()
    prefetch_chunks.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.selected == "native"
    assert not status.activated
    assert status.effective == "native"
    assert reason in status.fallback_reason


def test_admission_fallback_reuses_plan_without_second_prefetch(tmp_path, monkeypatch):
    checkpoint_dir, expected = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 0)
    discover_checkpoint_plan = loader._discover_checkpoint_plan

    with (
        mock.patch.object(
            loader,
            "_discover_checkpoint_plan",
            wraps=discover_checkpoint_plan,
        ) as discover,
        mock.patch.object(loader, "prefetch_files") as prefetch_files,
    ):
        weights = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    discover.assert_called_once()
    prefetch_files.assert_not_called()
    for name, tensor in expected.items():
        assert torch.equal(weights[name], tensor)
    status = loader.last_checkpoint_io_status
    assert status.selected == "native"
    assert status.effective == "native"
    assert "exceed effective host memory" in status.fallback_reason


def test_thread_start_failure_cleans_up_then_falls_back_once(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)
    native_weights = {"native": object()}

    with (
        mock.patch.object(
            weight_loader_module._RankStripedReadAheadSession,
            "start",
            side_effect=RuntimeError("thread start failed"),
        ),
        mock.patch.object(
            loader, "_load_weights_native", return_value=native_weights
        ) as native_load,
    ):
        weights = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert weights is native_weights
    native_load.assert_called_once()
    status = loader.last_checkpoint_io_status
    assert status.selected == "rank_striped_read_ahead"
    assert not status.activated
    assert status.effective == "native"
    assert "thread start failed" in status.fallback_reason


def test_peer_start_failure_cancels_reader_and_frees_communicator_once(tmp_path, monkeypatch):
    class _FakeNodeCommunicator:
        def __init__(self):
            self.free_calls = 0

        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1

        @staticmethod
        def allgather(value):
            return [value]

        @staticmethod
        def allreduce(value, op):
            del op
            return value

        def Free(self):
            self.free_calls += 1

    checkpoint_dir, _ = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    node_communicator = _FakeNodeCommunicator()
    read_started = threading.Event()
    read_cancelled = threading.Event()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)
    monkeypatch.setattr(loader, "_get_active_node_communicator", lambda: node_communicator)

    def coordinate_error(phase, error):
        assert error is None
        if phase == "rank-striped read-ahead start":
            return RuntimeError("Rank 1 failed to start its reader")
        return None

    def wait_for_cancel(local_chunks, max_workers, *, cancel_event, completion_callback):
        del local_chunks, max_workers, completion_callback
        read_started.set()
        assert cancel_event.wait(timeout=5)
        read_cancelled.set()

    monkeypatch.setattr(loader, "_coordinate_rank_error", coordinate_error)
    monkeypatch.setattr(loader, "_prefetch_chunks", wait_for_cancel)
    native_weights = {"native": object()}
    with mock.patch.object(
        loader, "_load_weights_native", return_value=native_weights
    ) as native_load:
        weights = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert read_started.is_set()
    assert read_cancelled.is_set()
    assert weights is native_weights
    native_load.assert_called_once()
    assert node_communicator.free_calls == 1


@pytest.mark.parametrize("failure", ["config", "layer_override"])
def test_source_or_configuration_error_is_not_retried_as_native(tmp_path, monkeypatch, failure):
    checkpoint_dir = tmp_path / failure
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()
    if failure == "config":
        (checkpoint_dir / "config.json").write_text("not-json")
    else:
        monkeypatch.setenv("TLLM_OVERRIDE_LAYER_NUM", "not-an-integer")

    loader = _rank_striped_loader()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)
    with (
        mock.patch.object(loader, "_load_weights_native") as native_load,
        pytest.raises(RuntimeError),
    ):
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    native_load.assert_not_called()


def test_rank_striped_session_overlaps_read_ahead_with_consumer(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    read_started = threading.Event()
    allow_read_to_finish = threading.Event()

    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)

    def block_background_read(local_chunks, max_workers, *, cancel_event, completion_callback):
        assert local_chunks
        assert max_workers > 0
        read_started.set()
        assert allow_read_to_finish.wait(timeout=5)
        if not cancel_event.is_set():
            for _, _, length in local_chunks:
                completion_callback(length)

    monkeypatch.setattr(loader, "_prefetch_chunks", block_background_read)

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as weights:
        try:
            assert read_started.wait(timeout=5)
            assert not allow_read_to_finish.is_set()
            assert "model.norm.weight" in weights
            assert loader.last_checkpoint_io_status.activated
        finally:
            allow_read_to_finish.set()

    status = loader.last_checkpoint_io_status
    assert status.effective == "rank_striped_read_ahead"
    assert status.completed_bytes == status.assigned_bytes


def test_safetensors_mapping_failure_cancels_reader_without_native_retry(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    read_cancelled = threading.Event()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)

    def wait_for_cancellation(local_chunks, max_workers, *, cancel_event, completion_callback):
        del local_chunks, max_workers, completion_callback
        assert cancel_event.wait(timeout=5)
        read_cancelled.set()

    monkeypatch.setattr(loader, "_prefetch_chunks", wait_for_cancellation)
    with (
        mock.patch.object(
            loader,
            "_load_weights_in_parallel",
            side_effect=ValueError("injected SafeTensors mapping failure"),
        ),
        mock.patch.object(loader, "_load_weights_native") as native_load,
        pytest.raises(RuntimeError, match="SafeTensors mapping"),
    ):
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert read_cancelled.is_set()
    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.activated
    assert status.effective == "none"
    assert "injected SafeTensors mapping failure" in status.fallback_reason


def test_materialization_failure_cancels_reader_without_native_retry(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    read_started = threading.Event()
    read_cancelled = threading.Event()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)

    def wait_for_cancellation(local_chunks, max_workers, *, cancel_event, completion_callback):
        del local_chunks, max_workers, completion_callback
        read_started.set()
        assert cancel_event.wait(timeout=5)
        read_cancelled.set()

    monkeypatch.setattr(loader, "_prefetch_chunks", wait_for_cancellation)
    native_map = loader._load_weights_in_parallel
    with (
        mock.patch.object(loader, "_load_weights_in_parallel", wraps=native_map) as map_weights,
        mock.patch.object(loader, "_load_weights_native") as native_load,
        pytest.raises(ValueError, match="materialization failed"),
    ):
        with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()):
            assert read_started.wait(timeout=5)
            raise ValueError("materialization failed")

    assert read_cancelled.is_set()
    assert map_weights.call_count == 1
    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.effective == "none"
    assert "materialization failed" in status.fallback_reason


def test_advisory_read_failure_keeps_materialized_weights_without_reload(tmp_path, monkeypatch):
    checkpoint_dir, expected = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)
    monkeypatch.setattr(
        loader,
        "_prefetch_chunks",
        mock.Mock(side_effect=OSError("injected read-ahead failure")),
    )

    native_map = loader._load_weights_in_parallel
    with mock.patch.object(loader, "_load_weights_in_parallel", wraps=native_map) as map_weights:
        weights = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert map_weights.call_count == 1
    for name, tensor in expected.items():
        assert torch.equal(weights[name], tensor)
    status = loader.last_checkpoint_io_status
    assert status.activated
    assert status.effective == "native"
    assert "injected read-ahead failure" in status.fallback_reason


def test_communicator_cleanup_failure_is_rank_coherent_and_not_retried(tmp_path, monkeypatch):
    class _FailingFreeCommunicator:
        @staticmethod
        def Get_rank():
            return 0

        @staticmethod
        def Get_size():
            return 1

        @staticmethod
        def allgather(value):
            return [value]

        @staticmethod
        def allreduce(value, op):
            del op
            return value

        @staticmethod
        def Free():
            raise OSError("injected communicator cleanup failure")

    checkpoint_dir, _ = _write_tiny_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    monkeypatch.setattr(loader, "_get_effective_available_host_memory", lambda: 1 << 40)
    monkeypatch.setattr(loader, "_get_active_node_communicator", _FailingFreeCommunicator)

    with (
        mock.patch.object(loader, "_load_weights_native") as native_load,
        pytest.raises(RuntimeError, match="communicator cleanup"),
    ):
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.activated
    assert status.effective == "none"
    assert "injected communicator cleanup failure" in status.fallback_reason


def test_native_and_rank_striped_outputs_match(tmp_path, monkeypatch):
    checkpoint_dir, expected = _write_tiny_checkpoint(tmp_path)
    native_loader = HfWeightLoader()
    striped_loader = _rank_striped_loader()
    monkeypatch.setattr(striped_loader, "_get_effective_available_host_memory", lambda: 1 << 40)

    native_weights = native_loader.load_weights(str(checkpoint_dir), mapping=Mapping())
    striped_weights = striped_loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert native_weights.keys() == striped_weights.keys() == expected.keys()
    for name, expected_tensor in expected.items():
        assert torch.equal(native_weights[name], expected_tensor)
        assert torch.equal(striped_weights[name], expected_tensor)


@pytest.mark.skipif(
    not weight_loader_module.ENABLE_MULTI_DEVICE or weight_loader_module.mpi_disabled(),
    reason="requires an MPI-enabled TensorRT-LLM build",
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_rank_striped_real_mpi_collectives_and_output_parity(tmp_path, mpi_pool_executor):
    checkpoint_dir, expected = _write_tiny_checkpoint(tmp_path)
    results = list(
        mpi_pool_executor.map(
            _run_rank_striped_mpi_smoke,
            [str(checkpoint_dir)] * mpi_pool_executor.num_workers,
        )
    )

    assert {result["rank"] for result in results} == {0, 1}
    assert all("checkpoint_io_policy must match" in result["policy_error"] for result in results)
    assert all("every mapping.world_size" in result["mapping_error"] for result in results)
    assert all(result["effective"] == "rank_striped_read_ahead" for result in results)
    assert all(result["assigned_bytes"] == result["completed_bytes"] for result in results)
    assert all(
        result["embed_shape"] == tuple(expected["model.embed_tokens.weight"].shape)
        for result in results
    )
    assert all(
        result["embed_sum"] == float(expected["model.embed_tokens.weight"].sum())
        for result in results
    )

    chunks = [chunk for result in results for chunk in result["chunks"]]
    assert len(chunks) == len(set(chunks))
    cursor = 0
    for file_name, offset, length in sorted(chunks, key=lambda chunk: chunk[1]):
        assert file_name == str(checkpoint_dir / "model.safetensors")
        assert offset == cursor
        cursor += length
    assert cursor == (checkpoint_dir / "model.safetensors").stat().st_size
