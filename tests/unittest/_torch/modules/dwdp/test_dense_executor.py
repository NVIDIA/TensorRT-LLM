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
"""``PyExecutor`` and the standalone dense DWDP manager: the teardown, the group, the transfer.

The executor holds the standalone ``DenseDwdpManager``, whose dense group is a tagged MPI
sub-communicator (``manager.standalone_group``) and which reserves the SM gather's SMs while it
is registered.  The executor is assembled with ``object.__new__`` and only the attributes a
method reads, as the executor tests do.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.modules.dwdp import fabric_gather
from tensorrt_llm._torch.modules.dwdp.dense import manager as dense_manager
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    STANDALONE_GROUP_TAG,
    DenseDwdpManager,
    StandaloneTransfer,
    dense_group,
    get_global_dense_dwdp_manager,
    standalone_group,
)
from tensorrt_llm._torch.modules.dwdp.dense.plan import DenseCopyLauncher
from tensorrt_llm._torch.modules.dwdp.dense.types import CopyMode, DenseMode, DenseSetupError
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only


def test_an_executor_built_without_init_holds_no_dense_manager() -> None:
    """Executors the tests assemble with ``object.__new__`` see no dense DWDP manager."""
    assert object.__new__(PyExecutor).dense_dwdp_manager is None


def test_shutdown_exits_the_dense_manager_once_after_the_kv_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``shutdown`` exits the dense manager the executor holds once, after the KV caches are shut
    down; the estimation executor, which holds none, leaves it to the final executor."""
    # ``_shutdown`` synchronizes the device when CUDA is available; this executor has none.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    log = []
    manager = MagicMock()
    manager.__exit__.side_effect = lambda *_args: log.append("dense.__exit__")
    executor = object.__new__(PyExecutor)
    executor.executor_request_queue = SimpleNamespace(enqueue_shutdown_request=lambda: None)
    executor.shutdown_event = SimpleNamespace(wait=lambda: None)
    executor.enable_attention_dp = True
    executor.kv_connector_manager = None
    executor._kv_connector_failed = False
    executor._profile_manager = SimpleNamespace(cleanup=lambda: None)
    executor.hang_detector = SimpleNamespace(detected=lambda: False)
    executor.worker_thread = SimpleNamespace(join=lambda: log.append("worker joined"))
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = lambda: None
    executor.sampler = SimpleNamespace()
    executor.worker_started = True
    executor.model_engine = SimpleNamespace(release_executor_resources=lambda **_kwargs: None)
    executor.draft_model_engine = None
    executor.resource_manager = SimpleNamespace(
        resource_managers={"kv": SimpleNamespace(shutdown=lambda: log.append("kv.shutdown"))}
    )
    executor.virtual_memory_pools = None
    executor.dwdp_manager = None
    executor.dense_dwdp_manager = manager

    executor.shutdown()

    assert log == ["worker joined", "kv.shutdown", "dense.__exit__"]
    manager.__exit__.assert_called_once_with(None, None, None)
    assert executor.dense_dwdp_manager is None


_ULYSSES_CP8 = Mapping(
    world_size=8, rank=5, cp_size=8, cp_config={"cp_type": CpType.ULYSSES}, moe_ep_size=8
)


@pytest.mark.parametrize(
    "mapping, tp_group",
    [
        (
            Mapping(world_size=4, rank=2, tp_size=4, moe_ep_size=4, enable_attention_dp=True),
            [0, 1, 2, 3],
        ),
        (
            Mapping(world_size=8, rank=5, tp_size=4, pp_size=2, enable_attention_dp=True),
            [4, 5, 6, 7],
        ),
        (_ULYSSES_CP8, list(range(8))),
    ],
    ids=["tp4", "second_pp_stage", "ulysses_cp8"],
)
def test_standalone_group_is_a_sub_communicator_over_the_dense_group(
    monkeypatch: pytest.MonkeyPatch, mapping: Mapping, tp_group: list[int]
) -> None:
    """The group is created with a tag of its own, not the default tag 0 of ``MPIDist.tp_comm``.

    ``MPIDist.tp_comm`` creates a group over the same ranks of the same parent, and MPI pairs the
    ``Create_group`` calls on one parent by their tag.
    """
    world = MagicMock()
    monkeypatch.setattr(dense_manager, "mpi_disabled", lambda: False)
    monkeypatch.setattr(dense_manager, "mpi_comm", lambda: world)

    group = standalone_group(mapping)

    assert list(dense_group(mapping).ranks) == tp_group
    world.group.Incl.assert_called_once_with(tp_group)
    world.Create_group.assert_called_once_with(
        world.group.Incl.return_value, tag=STANDALONE_GROUP_TAG
    )
    assert group is world.Create_group.return_value
    # Not the default tag, and a valid tag under every MPI (MPI_TAG_UB >= 32767).
    assert 0 < STANDALONE_GROUP_TAG <= 32767


def test_a_standalone_manager_needs_an_attention_dp_instance() -> None:
    with pytest.raises(ValueError, match="needs the mapping of an attention-DP instance"):
        DenseDwdpManager(DenseMode.STANDALONE, mapping=Mapping(world_size=4, tp_size=4, rank=0))


def test_the_dense_group_of_a_ulysses_instance_is_its_cp_group() -> None:
    """Ulysses CP ranks at TP 1 hold whole dense weights: the group is the CP group, ranked by
    CP rank; TP > 1 with CP, or attention DP with CP, has no standalone group."""
    group = dense_group(_ULYSSES_CP8)
    assert (group.ranks, group.rank, group.size, group.name) == (
        tuple(range(8)),
        5,
        8,
        "Ulysses CP",
    )
    DenseDwdpManager(DenseMode.STANDALONE, mapping=_ULYSSES_CP8)
    tp_and_cp = Mapping(
        world_size=8, rank=0, tp_size=2, cp_size=4, cp_config={"cp_type": CpType.ULYSSES}
    )
    assert dense_group(tp_and_cp) is None


# ---------------------------------------------------------------------------
# The standalone SM reserve and transfer
# ---------------------------------------------------------------------------


def _standalone_manager(copy: str | None = None) -> DenseDwdpManager:
    """A standalone manager of a 4-rank attention-DP instance with ``copy_mode`` ``copy``."""
    mapping = Mapping(world_size=4, rank=1, tp_size=4, moe_ep_size=4, enable_attention_dp=True)
    config = DenseConfig(enabled=True, copy=copy)
    return DenseDwdpManager(DenseMode.STANDALONE, mapping=mapping, config=config)


@pytest.fixture
def gather_blocks(monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """A resident gather of 30 blocks of 1024 threads; no reserve pinned."""
    monkeypatch.delenv("TRTLLM_CUTEDSL_SM_RESERVE", raising=False)
    monkeypatch.setenv("TRTLLM_DWDP_GATHER_BLOCKS", "30")
    monkeypatch.setattr(fabric_gather, "THREADS", 1024)
    yield 30
    assert sm_reserve() == 0


@pytest.mark.parametrize("copy, reserve", [("gather", 30), ("batched", 0), (None, 0)])
def test_a_standalone_gather_reserves_its_sms_from_registration_to_exit(
    gather_blocks: int, copy: str | None, reserve: int
) -> None:
    """The reserve is set before the model is built, so grids sized at construction see it; the
    batched copy-engine transfer reserves none."""
    manager = _standalone_manager(copy)
    assert sm_reserve() == 0
    manager.__enter__()
    try:
        assert get_global_dense_dwdp_manager() is manager
        assert sm_reserve() == reserve
    finally:
        manager.__exit__(None, None, None)
    assert sm_reserve() == 0
    assert get_global_dense_dwdp_manager() is None


@pytest.mark.parametrize("copy", ["batched", "per_slice", "gather"])
def test_a_pinned_reserve_holds_with_every_standalone_transfer(
    monkeypatch: pytest.MonkeyPatch, gather_blocks: int, copy: str
) -> None:
    """``TRTLLM_CUTEDSL_SM_RESERVE`` pins the reserve with the copy engines too, from
    registration to exit; with the gather it replaces the gather's count."""
    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", "12")
    with _standalone_manager(copy):
        assert sm_reserve() == 12
    assert sm_reserve() == 0


@pytest.mark.parametrize(
    "copy, mode", [(None, CopyMode.BATCHED), ("per_slice", CopyMode.PER_SLICE), ("gather", None)]
)
def test_standalone_setup_builds_the_transfer_of_its_copy_mode(
    monkeypatch: pytest.MonkeyPatch, gather_blocks: int, copy: str | None, mode: CopyMode | None
) -> None:
    """The transfer standalone setup builds follows ``copy_mode`` and never asks for the TMA
    kernel, which ``trtllm::dwdp_fabric_gather`` picks from ``TRTLLM_DWDP_GATHER_KERNEL``."""
    seen = []

    def setup_dense_dwdp(
        model: nn.Module, transfer: StandaloneTransfer, *args: object, **kwargs: object
    ) -> None:
        seen.append((transfer.mode, transfer.use_gather, transfer.use_tma))

    monkeypatch.setattr(dense_manager, "dense_roots_available", lambda model: True)
    monkeypatch.setattr(dense_manager, "setup_dense_dwdp", setup_dense_dwdp)
    expected = CopyMode.GATHER if copy == "gather" else mode
    with _standalone_manager(copy) as manager:
        assert manager.setup(nn.Module(), comm=MagicMock(), device_id=0) is None
    assert seen == [(expected, copy == "gather", False)]


def test_the_gather_launcher_refuses_a_kernel_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """``trtllm::dwdp_fabric_gather`` takes no kernel argument in this build."""
    monkeypatch.setattr(fabric_gather, "gather_op", lambda: object())
    with pytest.raises(DenseSetupError, match="takes no per-launch kernel argument"):
        DenseCopyLauncher({}, CopyMode.GATHER, device=torch.device("cpu"), use_tma=True)


def test_a_standalone_setup_refuses_a_model_without_dense_roots_before_its_group(
    monkeypatch: pytest.MonkeyPatch, gather_blocks: int
) -> None:
    """A model without dense roots raises before any collective; the reserve holds until exit."""

    def no_group(mapping: Mapping) -> None:
        pytest.fail("the standalone group was built for a model without dense roots")

    monkeypatch.setattr(dense_manager, "standalone_group", no_group)
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList()
    with _standalone_manager("gather") as manager:
        assert sm_reserve() == 30
        with pytest.raises(DenseSetupError, match="do not provide dense_dwdp_roots"):
            manager.setup(model)
        assert sm_reserve() == 30
        assert manager.dense is None
