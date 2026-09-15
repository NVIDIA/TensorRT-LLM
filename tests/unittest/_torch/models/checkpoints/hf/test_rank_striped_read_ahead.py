# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pickle
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints.checkpoint_catalog import (
    CheckpointCatalog,
    CheckpointExtent,
    CheckpointObject,
    CheckpointTensor,
)
from tensorrt_llm._torch.models.checkpoints.hf import rank_striped_read_ahead as read_ahead
from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as weight_loader_module
from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.weight_load_plan import (
    WeightDemand,
    WeightLoadOrderConfidence,
    WeightLoadPlan,
    WeightLoadPlanCoverage,
)
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
    monkeypatch.delenv("TRTLLM_DEMAND_ORDERED_READ_AHEAD_PHYSICAL_ORDER", raising=False)
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


def _demand_ordered_rank_striped_loader(*, partial_model_loading: bool = False) -> HfWeightLoader:
    return HfWeightLoader(
        checkpoint_io_policy="demand_ordered_rank_striped_read_ahead",
        partial_model_loading=partial_model_loading,
    )


class _TestWeightMapper:
    def __init__(self, mapping: Mapping, *, fail: bool = False) -> None:
        self._mapping = mapping
        self._fail = fail

    def build_weight_load_plan(self, catalog: CheckpointCatalog) -> WeightLoadPlan:
        if self._fail:
            raise RuntimeError("injected weight-plan failure")
        plan = WeightLoadPlan(
            catalog_id=catalog.catalog_id,
            rank=self._mapping.rank,
            world_size=self._mapping.world_size,
            coverage=WeightLoadPlanCoverage.CONSERVATIVE,
            ordering=WeightLoadOrderConfidence.ADVISORY,
            demands=(
                WeightDemand(
                    group_id="all_checkpoint_tensors",
                    source_names=tuple(tensor.name for tensor in catalog.tensors),
                    destination_ranks=(self._mapping.rank,),
                ),
            ),
        )
        plan.validate_against(catalog)
        return plan


def _test_weight_mapper(mapping: Mapping | None = None, *, fail: bool = False) -> _TestWeightMapper:
    return _TestWeightMapper(mapping if mapping is not None else Mapping(), fail=fail)


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
    log_info = mock.Mock()
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(loader, "open_weight_session", open_weight_session)
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", reader_start)
    monkeypatch.setattr(weight_loader_module.logger, "warning", warning)
    monkeypatch.setattr(weight_loader_module.logger, "info", log_info)
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
    assert any(
        "requested=rank_striped_read_ahead" in message
        and "selected=native" in message
        and "open_weight_session" in message
        for message, *_ in (call.args for call in log_info.call_args_list)
    )


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


def test_ordering_preserves_complete_fair_stripes(tmp_path, monkeypatch):
    monkeypatch.setattr(read_ahead, "_CHUNK_SIZE", 4)
    monkeypatch.setattr(read_ahead, "_READ_SIZE", 1)
    monkeypatch.setattr(read_ahead, "_WORKERS_PER_LOAD_GROUP", 3)
    monkeypatch.setattr(read_ahead, "_WORKERS_PER_RANK", 2)
    files = [(str(tmp_path / "a"), 10), (str(tmp_path / "b"), 7)]
    source_order = [
        read_ahead.ReadAheadExtent(str(tmp_path / "a"), 0, 4),
        read_ahead.ReadAheadExtent(str(tmp_path / "a"), 4, 4),
        read_ahead.ReadAheadExtent(str(tmp_path / "a"), 8, 2),
        read_ahead.ReadAheadExtent(str(tmp_path / "b"), 0, 4),
        read_ahead.ReadAheadExtent(str(tmp_path / "b"), 4, 3),
    ]
    plans = [
        read_ahead.build_local_plan(
            files,
            rank,
            4,
            ordered_extents=list(reversed(source_order)),
        )
        for rank in range(4)
    ]

    assert [plan.workers for plan in plans] == [1, 1, 1, 0]
    assert [extent for plan in plans for extent in plan.extents]
    assert set(extent for plan in plans for extent in plan.extents) == set(source_order)
    assert plans[0].extents[0] == source_order[-1]

    with pytest.raises(ValueError, match="complete permutation"):
        read_ahead.build_local_plan(
            files,
            0,
            4,
            ordered_extents=source_order[:-1],
        )


def test_weight_plans_compile_to_a_complete_node_demand_order(tmp_path, monkeypatch):
    monkeypatch.setattr(read_ahead, "_CHUNK_SIZE", 4)
    paths = (tmp_path / "a.safetensors", tmp_path / "b.safetensors")
    files = ((str(paths[0]), 12), (str(paths[1]), 8))
    catalog = CheckpointCatalog(
        objects=(
            CheckpointObject("a.safetensors", 12),
            CheckpointObject("b.safetensors", 8),
        ),
        tensors=(
            CheckpointTensor("late", extents=(CheckpointExtent("a.safetensors", 0, 4),)),
            CheckpointTensor("middle", extents=(CheckpointExtent("a.safetensors", 8, 4),)),
            CheckpointTensor("early", extents=(CheckpointExtent("b.safetensors", 4, 4),)),
        ),
    )

    def make_plan(rank, names, ordering=WeightLoadOrderConfidence.ADVISORY):
        return WeightLoadPlan(
            catalog_id=catalog.catalog_id,
            rank=rank,
            world_size=2,
            coverage=WeightLoadPlanCoverage.CONSERVATIVE,
            ordering=ordering,
            demands=tuple(
                WeightDemand(
                    # Lexical ordering intentionally differs from tuple order.
                    group_id=f"layer.{10 - index}",
                    source_names=(name,),
                    destination_ranks=(rank,),
                    priority=index,
                )
                for index, name in enumerate(names)
            ),
        )

    ordered = read_ahead.compile_weight_plan_extent_order(
        catalog,
        (
            make_plan(0, ("late", "middle", "early")),
            make_plan(1, ("early", "middle", "late")),
        ),
        files,
    )

    assert ordered == (
        read_ahead.ReadAheadExtent(str(paths[0]), 0, 4),
        read_ahead.ReadAheadExtent(str(paths[1]), 4, 4),
        read_ahead.ReadAheadExtent(str(paths[0]), 8, 4),
        read_ahead.ReadAheadExtent(str(paths[0]), 4, 4),
        read_ahead.ReadAheadExtent(str(paths[1]), 0, 4),
    )

    physical = read_ahead.compile_weight_plan_extent_order(
        catalog,
        (
            make_plan(
                0,
                ("early", "middle", "late"),
                WeightLoadOrderConfidence.OPAQUE,
            ),
            make_plan(
                1,
                ("late", "middle", "early"),
                WeightLoadOrderConfidence.OPAQUE,
            ),
        ),
        files,
    )
    assert physical == tuple(read_ahead._build_source_extents(files))

    dependency_plan = WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=1,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        ordering=WeightLoadOrderConfidence.ADVISORY,
        demands=(
            WeightDemand(
                group_id="dependency",
                source_names=("late", "middle"),
                destination_ranks=(0,),
                priority=100,
            ),
            WeightDemand(
                group_id="urgent-dependent",
                source_names=("early",),
                destination_ranks=(0,),
                priority=-100,
                predecessors=("dependency",),
            ),
        ),
    )
    predecessor_order = read_ahead.compile_weight_plan_extent_order(
        catalog, (dependency_plan,), files
    )
    assert predecessor_order[:3] == (
        read_ahead.ReadAheadExtent(str(paths[0]), 0, 4),
        read_ahead.ReadAheadExtent(str(paths[0]), 8, 4),
        read_ahead.ReadAheadExtent(str(paths[1]), 4, 4),
    )


def test_weight_plan_compilation_is_canonical_for_equal_plan_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(read_ahead, "_CHUNK_SIZE", 4)
    path = tmp_path / "model.safetensors"
    files = ((str(path), 12),)
    catalog = CheckpointCatalog(
        objects=(CheckpointObject(path.name, 12),),
        tensors=(
            CheckpointTensor("a", extents=(CheckpointExtent(path.name, 0, 4),)),
            CheckpointTensor("b", extents=(CheckpointExtent(path.name, 4, 4),)),
            CheckpointTensor("c", extents=(CheckpointExtent(path.name, 8, 4),)),
        ),
    )
    first = WeightDemand("first", ("b", "a"), (0,), priority=0)
    second = WeightDemand("second", ("c",), (0,), priority=0)

    def make_plan(demands):
        return WeightLoadPlan(
            catalog_id=catalog.catalog_id,
            rank=0,
            world_size=1,
            coverage=WeightLoadPlanCoverage.CONSERVATIVE,
            ordering=WeightLoadOrderConfidence.ADVISORY,
            demands=demands,
        )

    forward = make_plan((first, second))
    reordered = make_plan(
        (
            second,
            WeightDemand("first", ("a", "b"), (0,), priority=0),
        )
    )

    assert forward.plan_id == reordered.plan_id
    assert read_ahead.compile_weight_plan_extent_order(
        catalog, (forward,), files
    ) == read_ahead.compile_weight_plan_extent_order(catalog, (reordered,), files)

    # Equal-priority independent groups retain physical locality, regardless
    # of their group IDs, input order, or source-name ordering.
    assert read_ahead.compile_weight_plan_extent_order(catalog, (forward,), files) == tuple(
        read_ahead._build_source_extents(files)
    )


def test_physical_order_ablation_still_compiles_demand_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(read_ahead, "_CHUNK_SIZE", 4)
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"abcdefgh")
    files = ((str(path), 8),)
    catalog = CheckpointCatalog(
        objects=(CheckpointObject(path.name, 8),),
        tensors=(CheckpointTensor("early", extents=(CheckpointExtent(path.name, 4, 4),)),),
    )
    demand_plan = WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=1,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        ordering=WeightLoadOrderConfidence.ADVISORY,
        demands=(WeightDemand("early", ("early",), (0,)),),
    )
    original_plan = read_ahead.build_local_plan(files, 0, 1)
    compile_plan = mock.Mock(wraps=read_ahead.compile_weight_plan_extent_order)
    monkeypatch.setattr(weight_loader_module, "compile_weight_plan_extent_order", compile_plan)
    for physical_order in (False, True):
        monkeypatch.setenv(
            "TRTLLM_DEMAND_ORDERED_READ_AHEAD_PHYSICAL_ORDER", str(int(physical_order))
        )
        loader = _demand_ordered_rank_striped_loader()
        context = weight_loader_module._DemandReadAheadContext(
            None, None, catalog, list(files), 0, 1
        )
        loader._pending_read_ahead_session = context
        loader.activate_weight_session(
            weight_mapper=mock.Mock(build_weight_load_plan=mock.Mock(return_value=demand_plan))
        )
        assert context.reader is not None
        assert context.reader._plan.extents == (
            original_plan.extents if physical_order else tuple(reversed(original_plan.extents))
        )
        assert context.reader.cancel_and_close() is None
    assert compile_plan.call_count == 2


def test_read_ahead_handles_short_reads(tmp_path, monkeypatch):
    path = tmp_path / "weights"
    path.write_bytes(b"0123456789" * 20)
    monkeypatch.setattr(read_ahead, "_READ_SIZE", 40)
    extent = read_ahead.ReadAheadExtent(str(path), 0, 200)
    real_pread = read_ahead.os.pread

    def short_pread(file_descriptor, size, offset):
        return real_pread(file_descriptor, max(1, size // 2), offset)

    monkeypatch.setattr(read_ahead.os, "pread", short_pread)
    session = read_ahead.RankStripedReadAheadSession(
        None, None, read_ahead.ReadAheadPlan((extent,), 1)
    )
    try:
        assert session._read_extent(extent) == extent.length
    finally:
        assert session.cancel_and_close() is None


def test_deferred_session_issues_no_payload_io_before_release(tmp_path, monkeypatch):
    path = tmp_path / "weights"
    path.write_bytes(b"x")
    plan = read_ahead.ReadAheadPlan(
        (read_ahead.ReadAheadExtent(str(path), 0, 1),),
        workers=1,
    )
    pread = mock.Mock(return_value=b"x")
    monkeypatch.setattr(read_ahead.os, "pread", pread)
    session = read_ahead.RankStripedReadAheadSession(None, None, plan).start(defer_reads=True)

    time.sleep(0.01)
    pread.assert_not_called()
    session.release_reads()
    assert session._thread is not None
    session._thread.join(timeout=5)
    pread.assert_called_once()
    assert session.cancel_and_close() is None


def test_reader_open_failure_closes_already_opened_descriptors(
    tmp_path,
    monkeypatch,
):
    opened_path = tmp_path / "opened"
    failing_path = tmp_path / "failing"
    for path in (opened_path, failing_path):
        path.write_bytes(b"x")
    failed_plan = read_ahead.ReadAheadPlan(
        (
            read_ahead.ReadAheadExtent(str(opened_path), 0, 1),
            read_ahead.ReadAheadExtent(str(failing_path), 0, 1),
        ),
        workers=1,
    )
    real_open = read_ahead.os.open
    real_close = read_ahead.os.close
    newly_opened_descriptors = []
    closed_descriptors = []

    def fail_second_open(path, flags, *args, **kwargs):
        if path == str(failing_path):
            raise OSError("injected open failure")
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == str(opened_path):
            newly_opened_descriptors.append(descriptor)
        return descriptor

    def record_close(descriptor):
        closed_descriptors.append(descriptor)
        return real_close(descriptor)

    # os is process-wide: preserve unrelated opens and restore it before pytest
    # resumes capture/fixture teardown.
    with monkeypatch.context() as patch:
        patch.setattr(read_ahead.os, "open", fail_second_open)
        patch.setattr(read_ahead.os, "close", record_close)

        with pytest.raises(OSError, match="injected open failure"):
            read_ahead.RankStripedReadAheadSession(None, None, failed_plan)
        assert newly_opened_descriptors
        assert newly_opened_descriptors[0] in closed_descriptors
        with tempfile.TemporaryFile() as capture_file:
            capture_file.write(b"unrelated capture I/O")


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


@pytest.mark.parametrize(
    "requested_policy",
    ["auto", "rank_striped_read_ahead", "demand_ordered_rank_striped_read_ahead"],
)
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
        checkpoint_io_policy=(
            "rank_striped_read_ahead" if requested_policy == "auto" else requested_policy
        ),
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
    if requested_policy != "auto":
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
    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()):
        pass

    assert loader.last_checkpoint_io_status.activated


def test_demand_ordered_policy_requires_full_checkpoint_memory(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    monkeypatch.setattr(read_ahead, "_HOST_MEMORY_HEADROOM_BYTES", 0)
    monkeypatch.setattr(read_ahead, "_HOST_MEMORY_HEADROOM_FRACTION", 0)
    monkeypatch.setattr(weight_loader_module, "effective_available_host_memory", lambda: 1)
    loader = _demand_ordered_rank_striped_loader()
    start = mock.Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", start)

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()):
        pass

    assert loader.last_checkpoint_io_status.effective == "native"
    assert not loader.last_checkpoint_io_status.activated
    start.assert_not_called()


def test_reader_start_failure_cleans_up_and_degrades_without_reload(tmp_path, monkeypatch):
    checkpoint_dir, expected = _write_checkpoint(tmp_path)
    loader = _rank_striped_loader()
    native_load = mock.Mock(side_effect=AssertionError("must not reload"))
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

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as weights:
        pass

    assert torch.equal(weights["model.norm.weight"], expected["model.norm.weight"])
    assert close_file_descriptors
    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.selected == "rank_striped_read_ahead"
    assert not status.activated
    assert status.effective == "native"
    assert "start failed" in status.fallback_reason


def test_demand_ordered_reader_defers_activation_until_mapper_boundary(tmp_path, monkeypatch):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = _demand_ordered_rank_striped_loader()
    started = threading.Event()

    def wait_for_cancel(session, _extent):
        started.set()
        assert session._cancel.wait(timeout=5)
        return 0

    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "_read_extent", wait_for_cancel)
    reader_factory = mock.Mock(wraps=read_ahead.RankStripedReadAheadSession)
    monkeypatch.setattr(weight_loader_module, "RankStripedReadAheadSession", reader_factory)

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()):
        reader_factory.assert_not_called()
        assert not started.is_set()
        assert not loader.last_checkpoint_io_status.activated
        loader.activate_weight_session(weight_mapper=_test_weight_mapper())
        assert started.wait(timeout=5)
        assert loader.last_checkpoint_io_status.activated
        reader_factory.assert_called_once()

    assert loader.last_checkpoint_io_status.effective == ("demand_ordered_rank_striped_read_ahead")


@pytest.mark.parametrize("policy", ["native", "rank_striped_read_ahead"])
def test_existing_policies_do_not_plan_or_restart_on_activation(policy, monkeypatch):
    loader = HfWeightLoader(checkpoint_io_policy=policy)
    session = mock.Mock()
    mapper = mock.Mock()
    loader._pending_read_ahead_session = session
    consensus = mock.Mock(side_effect=AssertionError("must not coordinate"))
    monkeypatch.setattr(weight_loader_module, "coordinate_error", consensus)

    loader.activate_weight_session(weight_mapper=mapper)

    mapper.build_weight_load_plan.assert_not_called()
    session.start.assert_not_called()
    consensus.assert_not_called()


def test_mapper_readiness_failure_is_rank_coherent_before_start():
    communicator = mock.Mock()
    communicator.Get_size.return_value = 2
    communicator.allgather.return_value = [None, "ValueError: peer mapper failed"]
    local_errors = (None, ValueError("peer mapper failed"))
    raised_errors = []

    for local_error in local_errors:
        loader = _demand_ordered_rank_striped_loader()
        session = mock.Mock()
        session.active_communicator = communicator
        loader._pending_read_ahead_session = session

        with pytest.raises((RuntimeError, ValueError)) as raised:
            loader.activate_weight_session(local_error)
        raised_errors.append(raised.value)
        session.start.assert_not_called()
        assert not loader.last_checkpoint_io_status.activated
        assert "peer mapper failed" in loader.last_checkpoint_io_status.fallback_reason

    assert isinstance(raised_errors[0], RuntimeError)
    assert raised_errors[1] is local_errors[1]
    assert communicator.allgather.call_args_list == [
        mock.call(None),
        mock.call("ValueError: peer mapper failed"),
    ]


def test_demand_ordered_activation_orders_consensus_before_node_plan_exchange(monkeypatch):
    events = []
    catalog = CheckpointCatalog(
        objects=(CheckpointObject("model.safetensors", 1),),
        tensors=(
            CheckpointTensor("weight", extents=(CheckpointExtent("model.safetensors", 0, 1),)),
        ),
    )
    plan = WeightLoadPlan(
        catalog_id=catalog.catalog_id,
        rank=0,
        world_size=1,
        coverage=WeightLoadPlanCoverage.CONSERVATIVE,
        ordering=WeightLoadOrderConfidence.ADVISORY,
        demands=(WeightDemand("weight", ("weight",), (0,)),),
    )
    compiled_plan = read_ahead.ReadAheadPlan((), 0)
    mapper = mock.Mock()
    mapper.build_weight_load_plan.side_effect = lambda _catalog: (events.append("plan") or plan)
    node_communicator = mock.Mock()
    node_communicator.allgather.side_effect = lambda local_input: (
        events.append("node-allgather") or [local_input]
    )
    context = weight_loader_module._DemandReadAheadContext(
        None, node_communicator, catalog, [], 0, 1
    )
    compile_plan = mock.Mock(side_effect=lambda *_args: (events.append("compile") or compiled_plan))
    session = mock.Mock()
    reader_factory = mock.Mock(side_effect=lambda *_args: (events.append("construct") or session))
    session.start.side_effect = lambda **_kwargs: events.append("start")
    session.release_reads.side_effect = lambda: events.append("release")

    def record_consensus(_communicator, phase, error):
        assert error is None
        events.append(phase)
        return None

    monkeypatch.setattr(weight_loader_module, "coordinate_error", record_consensus)
    monkeypatch.setattr(weight_loader_module, "compile_weight_plan_extent_order", compile_plan)
    monkeypatch.setattr(
        weight_loader_module, "build_local_plan", mock.Mock(return_value=compiled_plan)
    )
    monkeypatch.setattr(weight_loader_module, "RankStripedReadAheadSession", reader_factory)
    loader = _demand_ordered_rank_striped_loader()
    loader._pending_read_ahead_session = context

    loader.activate_weight_session(weight_mapper=mapper)

    assert events == [
        "rank-striped mapper readiness",
        "plan",
        "demand-ordered weight plan build",
        "node-allgather",
        "compile",
        "construct",
        "demand-ordered reader setup",
        "start",
        "rank-striped reader activation",
        "release",
    ]


def test_physical_order_ablation_mismatch_disables_all_readers(monkeypatch):
    monkeypatch.delenv("TRTLLM_DEMAND_ORDERED_READ_AHEAD_PHYSICAL_ORDER", raising=False)
    catalog = CheckpointCatalog(objects=(), tensors=(CheckpointTensor("weight"),))
    mapper = _test_weight_mapper()
    plan = mapper.build_weight_load_plan(catalog)
    node_communicator = mock.Mock()
    node_communicator.allgather.return_value = [(plan, False), (plan, True)]
    context = weight_loader_module._DemandReadAheadContext(
        None, node_communicator, catalog, [], 0, 1
    )
    loader = _demand_ordered_rank_striped_loader()
    loader._pending_read_ahead_session = context

    loader.activate_weight_session(weight_mapper=mapper)

    assert context.reader is None
    assert not loader.last_checkpoint_io_status.activated
    assert "must agree across node-local ranks" in loader.last_checkpoint_io_status.fallback_reason


def test_unactivated_demand_session_reports_native(tmp_path):
    checkpoint_dir, expected = _write_checkpoint(tmp_path)
    loader = _demand_ordered_rank_striped_loader()

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as weights:
        assert torch.equal(weights["model.norm.weight"], expected["model.norm.weight"])

    status = loader.last_checkpoint_io_status
    assert not status.activated
    assert status.effective == "native"
    assert "not activated" in status.fallback_reason


def test_demand_ordered_activation_failure_degrades_collectively_without_reload(
    tmp_path, monkeypatch
):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = _demand_ordered_rank_striped_loader()
    native_load = mock.Mock(side_effect=AssertionError("must not reload"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(
        read_ahead.RankStripedReadAheadSession,
        "start",
        mock.Mock(side_effect=RuntimeError("activation failed")),
    )

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()):
        loader.activate_weight_session(weight_mapper=_test_weight_mapper())
        assert not loader.last_checkpoint_io_status.activated

    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.effective == "native"
    assert "activation failed" in status.fallback_reason


@pytest.mark.parametrize(
    ("failure_phase", "failure_message"),
    [
        ("plan", "injected weight-plan failure"),
        ("compiler", "injected compiler failure"),
        ("open", "injected descriptor-open failure"),
    ],
)
def test_demand_ordered_planning_failure_degrades_before_start_without_reload(
    tmp_path,
    monkeypatch,
    failure_phase,
    failure_message,
):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = _demand_ordered_rank_striped_loader()
    native_load = mock.Mock(side_effect=AssertionError("must not reload"))
    reader_start = mock.Mock(side_effect=AssertionError("must not start"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(read_ahead.RankStripedReadAheadSession, "start", reader_start)
    if failure_phase == "compiler":
        monkeypatch.setattr(
            weight_loader_module,
            "compile_weight_plan_extent_order",
            mock.Mock(side_effect=RuntimeError(failure_message)),
        )
    elif failure_phase == "open":
        monkeypatch.setattr(
            weight_loader_module,
            "RankStripedReadAheadSession",
            mock.Mock(side_effect=OSError(failure_message)),
        )

    mapper = _test_weight_mapper(fail=failure_phase == "plan")
    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as weights:
        assert weights
        loader.activate_weight_session(weight_mapper=mapper)
        assert not loader.last_checkpoint_io_status.activated

    reader_start.assert_not_called()
    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.effective == "native"
    assert failure_message in status.fallback_reason


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
        loader.activate_weight_session()
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

    with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as weights:
        pass

    assert torch.equal(weights["model.norm.weight"], expected["model.norm.weight"])
    native_load.assert_not_called()
    status = loader.last_checkpoint_io_status
    assert status.activated
    assert status.effective == "native"
    assert "injected read failure" in status.fallback_reason


@pytest.mark.parametrize(
    "loader_factory", [_rank_striped_loader, _demand_ordered_rank_striped_loader]
)
def test_mapping_and_materialization_failures_never_retry(tmp_path, monkeypatch, loader_factory):
    checkpoint_dir, _ = _write_checkpoint(tmp_path)
    loader = loader_factory()
    close_communicator = mock.Mock(wraps=weight_loader_module.close_node_communicator)
    monkeypatch.setattr(weight_loader_module, "close_node_communicator", close_communicator)
    native_load = mock.Mock(side_effect=AssertionError("must not reload"))
    monkeypatch.setattr(loader, "_load_weights_native", native_load)
    monkeypatch.setattr(
        loader, "_load_weights_in_parallel", mock.Mock(side_effect=RuntimeError("mapping failed"))
    )

    with pytest.raises(RuntimeError, match="mapping failed"):
        with loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()):
            pass
    native_load.assert_not_called()
    close_communicator.assert_called_once()

    loader = loader_factory()
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


def test_mapper_failure_closes_context_without_constructing_reader(monkeypatch):
    context = weight_loader_module._DemandReadAheadContext(
        None,
        mock.Mock(),
        CheckpointCatalog(objects=(), tensors=(CheckpointTensor("weight"),)),
        [],
        0,
        1,
    )
    loader = _demand_ordered_rank_striped_loader()
    monkeypatch.setattr(
        loader, "_start_rank_striped_read_ahead", lambda *_args, **_kwargs: ({}, context)
    )
    with pytest.raises(ValueError, match="mapper failed"):
        with loader.open_weight_session("checkpoint", mapping=Mapping()):
            loader.activate_weight_session(ValueError("mapper failed"))
    assert context.reader is None
    assert loader._pending_read_ahead_session is None
    context.node_communicator.Free.assert_called_once()


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
    striped_loader = _rank_striped_loader()
    with striped_loader.open_weight_session(str(checkpoint_dir), mapping=Mapping()) as striped:
        pass

    assert native_loader.last_checkpoint_io_status.effective == "native"
    assert striped_loader.last_checkpoint_io_status.effective == "rank_striped_read_ahead"
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
    with success_loader.open_weight_session(checkpoint_dir, mapping=mapping) as success_weights:
        pass
    success = success_loader.last_checkpoint_io_status.effective
    communicator.Barrier()

    demand_ordered_success_loader = _demand_ordered_rank_striped_loader()
    with demand_ordered_success_loader.open_weight_session(
        checkpoint_dir, mapping=mapping
    ) as demand_ordered_success_weights:
        demand_ordered_success_loader.activate_weight_session(
            weight_mapper=_test_weight_mapper(mapping)
        )
        assert demand_ordered_success_weights
    demand_ordered_success = demand_ordered_success_loader.last_checkpoint_io_status.effective
    communicator.Barrier()

    plan_failure_loader = _demand_ordered_rank_striped_loader()
    with plan_failure_loader.open_weight_session(
        checkpoint_dir, mapping=mapping
    ) as plan_failure_weights:
        plan_failure_loader.activate_weight_session(
            weight_mapper=_test_weight_mapper(mapping, fail=rank == world_size - 1)
        )
        assert plan_failure_weights
    plan_failure_status = plan_failure_loader.last_checkpoint_io_status
    communicator.Barrier()

    original_reader = weight_loader.RankStripedReadAheadSession
    if rank == world_size - 1:

        def fail_reader_open(*_args, **_kwargs):
            raise OSError("rank-local reader-open failure")

        weight_loader.RankStripedReadAheadSession = fail_reader_open
    open_failure_loader = _demand_ordered_rank_striped_loader()
    try:
        with open_failure_loader.open_weight_session(checkpoint_dir, mapping=mapping):
            open_failure_loader.activate_weight_session(weight_mapper=_test_weight_mapper(mapping))
        open_failure_status = open_failure_loader.last_checkpoint_io_status
    finally:
        weight_loader.RankStripedReadAheadSession = original_reader
    communicator.Barrier()

    readiness_loader = _demand_ordered_rank_striped_loader()
    local_readiness_error = (
        RuntimeError("rank-local readiness failure") if rank == world_size - 1 else None
    )
    readiness_error = None
    try:
        with readiness_loader.open_weight_session(
            checkpoint_dir, mapping=mapping
        ) as readiness_weights:
            assert readiness_weights
            readiness_loader.activate_weight_session(local_readiness_error)
    except BaseException as error:
        readiness_error = f"{type(error).__name__}: {error}"
    readiness_status = readiness_loader.last_checkpoint_io_status
    communicator.Barrier()

    original_start = rank_striped_read_ahead.RankStripedReadAheadSession.start
    if rank == world_size - 1:

        def fail_reader_start(_session, *, defer_reads=False):
            del defer_reads
            raise RuntimeError("rank-local demand-ordered reader-start failure")

        rank_striped_read_ahead.RankStripedReadAheadSession.start = fail_reader_start
    reader_start_loader = _demand_ordered_rank_striped_loader()
    try:
        with reader_start_loader.open_weight_session(
            checkpoint_dir, mapping=mapping
        ) as reader_start_weights:
            assert reader_start_weights
            reader_start_loader.activate_weight_session(weight_mapper=_test_weight_mapper(mapping))
        reader_start_status = reader_start_loader.last_checkpoint_io_status
    finally:
        rank_striped_read_ahead.RankStripedReadAheadSession.start = original_start
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
        with fallback_failure_loader.open_weight_session(
            checkpoint_dir, mapping=Mapping(world_size=1, rank=0, tp_size=1)
        ):
            pass
    except BaseException as error:
        fallback_failure_error = str(error)
    communicator.Barrier()

    subgroup = communicator.Split(color=rank, key=0)
    set_thread_local_mpi_comm(subgroup)
    subgroup_error = None
    try:
        subgroup_loader = _rank_striped_loader(partial_model_loading=rank == 0)
        with subgroup_loader.open_weight_session(
            checkpoint_dir, mapping=Mapping(world_size=1, rank=0, tp_size=1)
        ) as subgroup_weights:
            pass
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
            body_loader.activate_weight_session()
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
            read_loader.activate_weight_session()
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
        "demand_ordered_success": demand_ordered_success,
        "demand_ordered_sum": float(demand_ordered_success_weights["model.norm.weight"].sum()),
        "plan_failure_effective": plan_failure_status.effective,
        "plan_failure_reason": plan_failure_status.fallback_reason,
        "plan_failure_sum": float(plan_failure_weights["model.norm.weight"].sum()),
        "open_failure_effective": open_failure_status.effective,
        "open_failure_reason": open_failure_status.fallback_reason,
        "readiness_error": readiness_error,
        "readiness_effective": readiness_status.effective,
        "readiness_reason": readiness_status.fallback_reason,
        "reader_start_effective": reader_start_status.effective,
        "reader_start_reason": reader_start_status.fallback_reason,
        "reader_start_sum": float(reader_start_weights["model.norm.weight"].sum()),
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
        result["demand_ordered_success"] == "demand_ordered_rank_striped_read_ahead"
        for result in results
    )
    assert all(result["demand_ordered_sum"] == expected_sum for result in results)
    assert all(result["plan_failure_effective"] == "native" for result in results)
    assert all(
        "injected weight-plan failure" in result["plan_failure_reason"] for result in results
    )
    assert all(result["plan_failure_sum"] == expected_sum for result in results)
    assert all(result["open_failure_effective"] == "native" for result in results)
    assert all(
        "rank-local reader-open failure" in result["open_failure_reason"] for result in results
    )
    assert all("rank-local readiness failure" in result["readiness_error"] for result in results)
    assert all(result["readiness_effective"] == "none" for result in results)
    assert all("rank-local readiness failure" in result["readiness_reason"] for result in results)
    assert all(result["reader_start_effective"] == "native" for result in results)
    assert all(
        "rank-local demand-ordered reader-start failure" in result["reader_start_reason"]
        for result in results
    )
    assert all(result["reader_start_sum"] == expected_sum for result in results)
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
