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
"""``PyExecutor`` and the DWDP managers: which one primes, which ones it accepts, the teardown.

With expert DWDP the executor holds the ``DwdpManager``, which drives the attached dense DWDP
manager; with standalone dense DWDP it holds the ``DenseDwdpManager`` itself, whose dense group is
a tagged MPI sub-communicator (``manager.standalone_group``) and which reserves the SM gather's
SMs while it is registered.  The executor is assembled with ``object.__new__`` and only the
attributes a method reads, as the executor tests do.
"""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torch import nn

from tensorrt_llm._torch.cute_dsl_utils import sm_reserve
from tensorrt_llm._torch.modules.dwdp import fabric_gather
from tensorrt_llm._torch.modules.dwdp.dense import manager as dense_manager
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    STANDALONE_GROUP_TAG,
    DenseDwdpManager,
    get_global_dense_dwdp_manager,
    standalone_group,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode, DenseSetupError
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


def test_prime_dwdp_primes_a_standalone_dense_manager() -> None:
    executor = object.__new__(PyExecutor)
    executor.dwdp_manager = None
    executor.dense_dwdp_manager = MagicMock()

    executor._prime_dwdp()

    executor.dense_dwdp_manager.prefetch_first_layers.assert_called_once_with()


def test_prime_dwdp_starts_expert_dwdp_which_drives_the_attached_dense_manager() -> None:
    executor = object.__new__(PyExecutor)
    executor.dwdp_manager = MagicMock()
    executor.dense_dwdp_manager = None

    executor._prime_dwdp()

    executor.dwdp_manager.start_forward.assert_called_once_with()


@pytest.mark.parametrize(
    "expert, dense, disable_overlap_scheduler, error",
    [
        (False, True, True, None),
        (False, True, False, None),
        (True, False, False, "DWDP requires disable_overlap_scheduler=True. Overlap scheduler"),
        (True, True, True, "pass dense_dwdp_manager only without DWDP"),
    ],
    ids=["standalone-no_overlap", "standalone-overlap", "expert-overlap", "both"],
)
def test_the_executor_accepts_a_standalone_dense_manager_with_either_loop(
    expert: bool, dense: bool, disable_overlap_scheduler: bool, error: str | None
) -> None:
    """Expert DWDP still refuses the overlap scheduler, and drives dense DWDP itself."""
    managers = (MagicMock() if expert else None, MagicMock() if dense else None)
    if error is None:
        PyExecutor._validate_dwdp_managers(*managers, disable_overlap_scheduler)
    else:
        with pytest.raises(ValueError, match=error):
            PyExecutor._validate_dwdp_managers(*managers, disable_overlap_scheduler)


@pytest.mark.parametrize("mode", ["standalone", "attached"])
def test_shutdown_exits_the_manager_it_holds_once_after_the_kv_caches(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """``shutdown`` exits the manager the executor holds once, after the KV caches are shut down.

    The executor still holds the model engine then: it drops the engine only once its execution
    resources are released. ``release_resources``, which ends the profiling phase, exits the
    managers the same way, so the creator binds them to the serving phase only.
    """
    # ``shutdown`` synchronizes the device when CUDA is available; this executor has none.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    log = []
    executor = object.__new__(PyExecutor)
    manager = MagicMock()
    manager.__exit__.side_effect = lambda *_args: log.append(
        (f"{mode}.__exit__", hasattr(executor, "model_engine"))
    )
    executor._closed = False
    executor._resources_released = False
    executor.worker_started = True
    executor.executor_request_queue = SimpleNamespace(enqueue_shutdown_request=lambda: None)
    executor.shutdown_event = SimpleNamespace(wait=lambda: None)
    executor._profile_manager = SimpleNamespace(cleanup=lambda: None)
    executor.hang_detector = SimpleNamespace(detected=lambda: False)
    executor.worker_thread = SimpleNamespace(join=lambda: log.append("worker joined"))
    executor.enable_attention_dp = False
    executor.kv_connector_manager = None
    executor._kv_connector_failed = False
    executor.dist = SimpleNamespace(pp_size=1)
    executor._shutdown_sleep_wakeup_listeners = lambda: None
    executor.model_engine = SimpleNamespace(release_executor_resources=lambda **_kwargs: None)
    executor.draft_model_engine = None
    executor.kv_cache_transceiver = None
    executor.resource_manager = SimpleNamespace(
        resource_managers={"kv": SimpleNamespace(shutdown=lambda: log.append("kv.shutdown"))}
    )
    executor.virtual_memory_pools = None
    executor.sampler = SimpleNamespace()
    executor.dwdp_manager = manager if mode == "attached" else None
    executor.dense_dwdp_manager = manager if mode == "standalone" else None

    executor.shutdown()
    executor.shutdown()

    assert log == ["worker joined", "kv.shutdown", (f"{mode}.__exit__", True)]
    manager.__exit__.assert_called_once_with(None, None, None)
    assert executor.dwdp_manager is None and executor.dense_dwdp_manager is None
    assert executor._closed and not hasattr(executor, "model_engine")


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
    ],
    ids=["tp4", "second_pp_stage"],
)
def test_standalone_group_is_a_sub_communicator_over_the_tp_group(
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

    assert mapping.tp_group == tp_group
    world.group.Incl.assert_called_once_with(tp_group)
    world.Create_group.assert_called_once_with(
        world.group.Incl.return_value, tag=STANDALONE_GROUP_TAG
    )
    assert group is world.Create_group.return_value
    # Not the default tag, and a valid tag under every MPI (MPI_TAG_UB >= 32767).
    assert 0 < STANDALONE_GROUP_TAG <= 32767


# ---------------------------------------------------------------------------
# The standalone SM reserve
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
    """The reserve is set before the model is built, so grids sized at construction see it."""
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
    """``TRTLLM_CUTEDSL_SM_RESERVE`` pins the reserve with the copy engines too, as it does for
    ``DwdpManager``, from registration to exit; with the gather it replaces the gather's count."""
    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", "12")
    with _standalone_manager(copy):
        assert sm_reserve() == 12
    assert sm_reserve() == 0


def test_an_attached_dense_manager_leaves_the_reserve_to_dwdp_manager(gather_blocks: int) -> None:
    with DenseDwdpManager(DenseMode.ATTACHED):
        assert sm_reserve() == 0


def test_an_attached_dense_manager_takes_no_config() -> None:
    """With expert DWDP, DwdpManager passes the settings to dense setup itself."""
    with pytest.raises(ValueError, match="an attached DenseDwdpManager takes no mapping and no"):
        DenseDwdpManager(DenseMode.ATTACHED, config=DenseConfig(enabled=True))


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
