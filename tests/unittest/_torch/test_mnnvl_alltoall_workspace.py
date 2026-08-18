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

import gc
from types import SimpleNamespace
from unittest.mock import Mock
from weakref import WeakSet

import pytest
import torch

import tensorrt_llm._mnnvl_utils as mnnvl
import tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided as one_sided_module
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm._torch.mnnvl_alltoall_workspace import _MnnvlAlltoAllWorkspaceLifecycle
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
from tensorrt_llm._torch.modules.fused_moe.communication.nvlink_two_sided import NVLinkTwoSided


class _Client:
    def __init__(self, *, idle: bool = True) -> None:
        self.idle = idle
        self.reset_count = 0

    def _mnnvl_checkpoint_is_idle(self) -> bool:
        return self.idle

    def _mnnvl_checkpoint_reset(self) -> None:
        self.reset_count += 1


class _FailingResetClient(_Client):
    def _mnnvl_checkpoint_reset(self) -> None:
        raise RuntimeError("frontend reset failed")


class _FakeComm:
    def __init__(
        self,
        clients_idle_by_rank: list[bool] | None = None,
        gathered_values: list[list[bool]] | None = None,
    ) -> None:
        self.barrier_count = 0
        self.allgather_count = 0
        self.clients_idle_by_rank = clients_idle_by_rank
        self.gathered_values = list(gathered_values or [])

    def barrier(self) -> None:
        self.barrier_count += 1

    def allgather(self, local_clients_idle: bool) -> list[bool]:
        self.allgather_count += 1
        if self.gathered_values:
            return self.gathered_values.pop(0)
        if self.clients_idle_by_rank is not None:
            return self.clients_idle_by_rank
        return [local_clients_idle, local_clients_idle]


def _make_lifecycle() -> tuple[_MnnvlAlltoAllWorkspaceLifecycle, Mock, torch.Tensor]:
    workspace_state = {}
    memory = Mock(mapped=True)
    memory.comm = _FakeComm()
    workspace = torch.zeros(1, dtype=torch.uint8)
    metainfo = torch.tensor([1])
    lifecycle = _MnnvlAlltoAllWorkspaceLifecycle.get_or_create(
        workspace_state=workspace_state,
        memory=memory,
        workspace=workspace,
        metainfo=metainfo,
        metainfo_index={
            "FLAG_VAL_OFFSET_INDEX": 0,
            "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 0,
            "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 0,
        },
        ep_rank=0,
        ep_size=2,
        health=None,
    )
    return lifecycle, memory, metainfo


def _register_without_watchdog(
    lifecycle: _MnnvlAlltoAllWorkspaceLifecycle,
    client: _Client,
) -> None:
    lifecycle.register(
        client,
        watchdog_timeout_s=None,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )


def test_checkpoint_prepare_rejects_any_active_shared_client() -> None:
    lifecycle, memory, _ = _make_lifecycle()
    idle = _Client()
    _register_without_watchdog(lifecycle, idle)
    active = _Client(idle=False)
    _register_without_watchdog(lifecycle, active)

    with pytest.raises(RuntimeError, match="active MoE All-to-All phase"):
        lifecycle.checkpoint_prepare()

    memory.checkpoint_prepare.assert_not_called()


def test_repeated_checkpoint_prepare_skips_shared_preflight() -> None:
    lifecycle, memory, _ = _make_lifecycle()
    memory.mapped = False

    lifecycle.checkpoint_prepare()

    assert memory.comm.allgather_count == 0
    memory.checkpoint_prepare.assert_called_once_with()


def test_detached_checkpoint_prepare_stops_stale_watchdog() -> None:
    lifecycle, memory, _ = _make_lifecycle()
    coordinator = Mock()
    watchdog = Mock()
    coordinator.acquire_watchdog.return_value = watchdog
    lifecycle._coordinator = coordinator
    lifecycle.register(
        _Client(),
        watchdog_timeout_s=5.0,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )
    memory.mapped = False

    lifecycle.checkpoint_prepare()

    coordinator.release_watchdog.assert_called_once_with(watchdog)
    assert memory.comm.allgather_count == 0
    memory.checkpoint_prepare.assert_called_once_with()


def test_checkpoint_prepare_rejects_uninitialized_communicator() -> None:
    lifecycle, memory, _ = _make_lifecycle()
    memory.comm = None

    with pytest.raises(RuntimeError, match="communicator is not initialized"):
        lifecycle.checkpoint_prepare()

    memory.checkpoint_prepare.assert_not_called()


def test_checkpoint_prepare_rejects_communicator_size_mismatch() -> None:
    lifecycle, memory, _ = _make_lifecycle()
    memory.comm = _FakeComm(clients_idle_by_rank=[True])

    with pytest.raises(RuntimeError, match="communicator size does not match"):
        lifecycle.checkpoint_prepare()

    memory.checkpoint_prepare.assert_not_called()


def test_checkpoint_prepare_timeout_fails_closed(monkeypatch) -> None:
    lifecycle, memory, _ = _make_lifecycle()
    request = SimpleNamespace(
        test=lambda: (False, None),
        Cancel=Mock(),
        Free=Mock(),
    )
    memory.comm = SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 2,
        irecv=lambda source, tag: request,
        isend=lambda value, dest, tag: SimpleNamespace(test=lambda: (True, None)),
    )
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S", 0.0)

    with pytest.raises(TimeoutError, match="workspace idle readiness"):
        lifecycle.checkpoint_prepare()

    memory.checkpoint_fail_closed.assert_called_once_with()
    memory.checkpoint_prepare.assert_not_called()


def test_checkpoint_prepare_rejects_remote_active_client_before_watchdog_stop() -> None:
    lifecycle, memory, _ = _make_lifecycle()
    memory.comm = _FakeComm(clients_idle_by_rank=[True, False])
    coordinator = Mock()
    watchdog = Mock()
    coordinator.acquire_watchdog.return_value = watchdog
    lifecycle._coordinator = coordinator
    client = _Client()
    lifecycle.register(
        client,
        watchdog_timeout_s=5.0,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )

    with pytest.raises(RuntimeError, match=r"active MoE All-to-All phase on ranks \[1\]"):
        lifecycle.checkpoint_prepare()

    coordinator.release_watchdog.assert_not_called()
    memory.checkpoint_prepare.assert_not_called()


def test_checkpoint_suspends_and_recreates_one_shared_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, memory, metainfo = _make_lifecycle()
    old_coordinator = Mock()
    old_watchdog = Mock()
    old_coordinator.acquire_watchdog.return_value = old_watchdog
    lifecycle._coordinator = old_coordinator
    first = _Client()
    second = _Client()

    for client in (first, second):
        lifecycle.register(
            client,
            watchdog_timeout_s=5.0,
            watchdog_poll_interval_s=0.1,
            watchdog_on_timeout=None,
        )

    old_coordinator.acquire_watchdog.assert_called_once_with(
        ep_size=2,
        timeout_s=5.0,
        poll_interval_s=0.1,
        on_timeout=None,
    )
    assert lifecycle.watchdog_for(first) is old_watchdog
    assert lifecycle.watchdog_for(second) is old_watchdog

    lifecycle.checkpoint_prepare()

    old_coordinator.release_watchdog.assert_called_once_with(old_watchdog)
    memory.checkpoint_prepare.assert_called_once_with()

    memory.mapped = False
    memory.checkpoint_restore.return_value = True
    new_coordinator = Mock()
    new_watchdog = Mock()
    new_coordinator.acquire_watchdog.return_value = new_watchdog
    monkeypatch.setattr(
        lifecycle,
        "_create_coordinator",
        Mock(return_value=new_coordinator),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    comm = _FakeComm()

    lifecycle.checkpoint_restore(comm, Mock(return_value=metainfo))

    memory._checkpoint_restore_complete.assert_called_once_with()
    new_coordinator.acquire_watchdog.assert_called_once_with(
        ep_size=2,
        timeout_s=5.0,
        poll_interval_s=0.1,
        on_timeout=None,
    )
    assert lifecycle.watchdog_for(first) is new_watchdog
    assert lifecycle.watchdog_for(second) is new_watchdog
    assert first.reset_count == 1
    assert second.reset_count == 1
    assert comm.allgather_count == 1


def test_unregister_stops_watchdog_after_last_enabled_client() -> None:
    lifecycle, _, _ = _make_lifecycle()
    coordinator = Mock()
    watchdog = Mock()
    coordinator.acquire_watchdog.return_value = watchdog
    lifecycle._coordinator = coordinator
    first = _Client()
    second = _Client()
    for client in (first, second):
        lifecycle.register(
            client,
            watchdog_timeout_s=5.0,
            watchdog_poll_interval_s=0.1,
            watchdog_on_timeout=None,
        )

    lifecycle.unregister(first)
    coordinator.release_watchdog.assert_not_called()

    lifecycle.unregister(second)
    coordinator.release_watchdog.assert_called_once_with(watchdog)


def test_shared_watchdog_configuration_mismatch_rejects_new_client() -> None:
    lifecycle, _, _ = _make_lifecycle()
    coordinator = Mock()
    coordinator.acquire_watchdog.return_value = Mock()
    lifecycle._coordinator = coordinator
    first = _Client()
    second = _Client()
    lifecycle.register(
        first,
        watchdog_timeout_s=5.0,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )

    with pytest.raises(ValueError, match="same watchdog configuration"):
        lifecycle.register(
            second,
            watchdog_timeout_s=10.0,
            watchdog_poll_interval_s=0.1,
            watchdog_on_timeout=None,
        )

    assert lifecycle.watchdog_for(second) is None
    coordinator.acquire_watchdog.assert_called_once()


def test_watchdog_registration_is_deferred_while_workspace_is_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, memory, metainfo = _make_lifecycle()
    memory.mapped = False
    memory.checkpoint_restore.return_value = True
    old_coordinator = Mock()
    lifecycle._coordinator = old_coordinator
    client = _Client()

    lifecycle.register(
        client,
        watchdog_timeout_s=5.0,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )

    old_coordinator.acquire_watchdog.assert_not_called()
    assert lifecycle.watchdog_for(client) is None

    new_coordinator = Mock()
    new_watchdog = Mock()
    new_coordinator.acquire_watchdog.return_value = new_watchdog
    monkeypatch.setattr(
        lifecycle,
        "_create_coordinator",
        Mock(return_value=new_coordinator),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())

    lifecycle.checkpoint_restore(_FakeComm(), Mock(return_value=metainfo))

    assert lifecycle.watchdog_for(client) is new_watchdog


def test_checkpoint_restore_failure_before_watchdog_start_stays_unpublished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, memory, _ = _make_lifecycle()
    client = _Client()
    _register_without_watchdog(lifecycle, client)
    memory.mapped = False
    memory.checkpoint_restore.return_value = True
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())

    with pytest.raises(RuntimeError, match="frontend restore failed"):
        lifecycle.checkpoint_restore(
            _FakeComm(),
            Mock(side_effect=RuntimeError("frontend restore failed")),
        )

    memory._checkpoint_restore_failed.assert_called_once_with()
    memory._checkpoint_restore_complete.assert_not_called()
    assert client.reset_count == 0


def test_checkpoint_restore_rejects_changed_metainfo_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, memory, _ = _make_lifecycle()
    client = _Client()
    _register_without_watchdog(lifecycle, client)
    memory.mapped = False
    memory.checkpoint_restore.return_value = True
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())

    with pytest.raises(RuntimeError, match="metainfo changed"):
        lifecycle.checkpoint_restore(
            _FakeComm(),
            Mock(return_value=torch.tensor([2])),
        )

    memory._checkpoint_restore_failed.assert_called_once_with()
    memory._checkpoint_restore_complete.assert_not_called()
    assert client.reset_count == 0


def test_checkpoint_restore_failure_after_watchdog_start_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, memory, metainfo = _make_lifecycle()
    old_coordinator = Mock()
    old_watchdog = Mock()
    old_coordinator.acquire_watchdog.return_value = old_watchdog
    lifecycle._coordinator = old_coordinator
    client = _FailingResetClient()
    lifecycle.register(
        client,
        watchdog_timeout_s=5.0,
        watchdog_poll_interval_s=0.1,
        watchdog_on_timeout=None,
    )
    lifecycle.checkpoint_prepare()
    memory.mapped = False
    memory.checkpoint_restore.return_value = True
    new_coordinator = Mock()
    new_watchdog = Mock()
    new_coordinator.acquire_watchdog.return_value = new_watchdog
    monkeypatch.setattr(
        lifecycle,
        "_create_coordinator",
        Mock(return_value=new_coordinator),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())

    with pytest.raises(RuntimeError, match="frontend reset failed"):
        lifecycle.checkpoint_restore(_FakeComm(), Mock(return_value=metainfo))

    new_coordinator.release_watchdog.assert_called_once_with(new_watchdog)
    memory._checkpoint_restore_failed.assert_called_once_with()
    memory._checkpoint_restore_complete.assert_not_called()


def test_checkpoint_restore_remote_failure_fails_closed_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, memory, metainfo = _make_lifecycle()
    client = _Client()
    _register_without_watchdog(lifecycle, client)
    memory.mapped = False
    memory.checkpoint_restore.return_value = True
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    comm = _FakeComm(gathered_values=[[True, False]])

    with pytest.raises(RuntimeError, match=r"restore failed on ranks \[1\]"):
        lifecycle.checkpoint_restore(comm, Mock(return_value=metainfo))

    memory._checkpoint_restore_failed.assert_called_once_with()
    memory._checkpoint_restore_complete.assert_not_called()


def test_checkpoint_restore_reports_local_failure_to_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, memory, _ = _make_lifecycle()
    memory.mapped = False
    memory.checkpoint_restore.return_value = True
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    comm = _FakeComm(gathered_values=[[False, True]])

    with pytest.raises(RuntimeError, match="frontend restore failed"):
        lifecycle.checkpoint_restore(
            comm,
            Mock(side_effect=RuntimeError("frontend restore failed")),
        )

    assert comm.allgather_count == 1
    memory._checkpoint_restore_failed.assert_called_once_with()


def test_two_sided_checkpoint_prepare_rejects_active_shared_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = WeakSet()
    monkeypatch.setattr(NVLinkTwoSided, "_INSTANCES", instances)
    checkpoint_prepare = Mock()
    monkeypatch.setattr(mnnvl.MnnvlMoe, "checkpoint_prepare", checkpoint_prepare)
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_workspace",
        Mock(comm=_FakeComm()),
    )
    idle = NVLinkTwoSided.__new__(NVLinkTwoSided)
    idle.ep_size = 2
    idle._dispatch_state = {}
    active = NVLinkTwoSided.__new__(NVLinkTwoSided)
    active._dispatch_state = {"alltoall_info": object()}
    instances.update((idle, active))

    with pytest.raises(
        RuntimeError,
        match=r"active MoE All-to-All phase on ranks \[0, 1\]",
    ):
        idle.checkpoint_prepare()

    checkpoint_prepare.assert_not_called()


def test_two_sided_repeated_checkpoint_prepare_skips_shared_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = WeakSet()
    monkeypatch.setattr(NVLinkTwoSided, "_INSTANCES", instances)
    checkpoint_prepare = Mock()
    monkeypatch.setattr(mnnvl.MnnvlMoe, "checkpoint_prepare", checkpoint_prepare)
    comm = _FakeComm()
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_workspace",
        Mock(mapped=False, comm=comm),
    )
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_prepare_workspace",
        Mock(mapped=False),
    )
    owner = NVLinkTwoSided.__new__(NVLinkTwoSided)
    owner.ep_size = 2
    owner._dispatch_state = {}
    instances.add(owner)

    owner.checkpoint_prepare()

    assert comm.allgather_count == 0
    checkpoint_prepare.assert_called_once_with()


def test_two_sided_checkpoint_prepare_rejects_uninitialized_communicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = WeakSet()
    monkeypatch.setattr(NVLinkTwoSided, "_INSTANCES", instances)
    checkpoint_prepare = Mock()
    monkeypatch.setattr(mnnvl.MnnvlMoe, "checkpoint_prepare", checkpoint_prepare)
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_workspace",
        Mock(mapped=True, comm=None),
    )
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_prepare_workspace",
        Mock(mapped=True),
    )
    owner = NVLinkTwoSided.__new__(NVLinkTwoSided)
    owner.ep_size = 2
    owner._dispatch_state = {}
    instances.add(owner)

    with pytest.raises(RuntimeError, match="communicator is not initialized"):
        owner.checkpoint_prepare()

    checkpoint_prepare.assert_not_called()


def test_two_sided_checkpoint_prepare_timeout_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = WeakSet()
    monkeypatch.setattr(NVLinkTwoSided, "_INSTANCES", instances)
    request = SimpleNamespace(
        test=lambda: (False, None),
        Cancel=Mock(),
        Free=Mock(),
    )
    comm = SimpleNamespace(
        Get_rank=lambda: 0,
        Get_size=lambda: 2,
        irecv=lambda source, tag: request,
        isend=lambda value, dest, tag: SimpleNamespace(test=lambda: (True, None)),
    )
    main_workspace = Mock(mapped=True, comm=comm)
    prepare_workspace = Mock(mapped=True)
    monkeypatch.setattr(mnnvl.MnnvlMoe, "moe_workspace", main_workspace)
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_prepare_workspace",
        prepare_workspace,
    )
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(mnnvl, "_MNNVL_CHECKPOINT_COLLECTIVE_POLL_INTERVAL_S", 0.0)
    owner = NVLinkTwoSided.__new__(NVLinkTwoSided)
    owner.ep_size = 2
    owner._dispatch_state = {}
    instances.add(owner)

    with pytest.raises(TimeoutError, match="workspace idle readiness"):
        owner.checkpoint_prepare()

    main_workspace.checkpoint_fail_closed.assert_called_once_with()
    prepare_workspace.checkpoint_fail_closed.assert_called_once_with()


def test_two_sided_checkpoint_restore_resets_all_shared_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = WeakSet()
    monkeypatch.setattr(NVLinkTwoSided, "_INSTANCES", instances)
    checkpoint_restore = Mock()
    monkeypatch.setattr(mnnvl.MnnvlMoe, "checkpoint_restore", checkpoint_restore)
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_workspace",
        Mock(mapped=False),
    )
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_prepare_workspace",
        Mock(mapped=False),
    )
    first = NVLinkTwoSided.__new__(NVLinkTwoSided)
    first._dispatch_state = {"alltoall_info": object()}
    second = NVLinkTwoSided.__new__(NVLinkTwoSided)
    second._dispatch_state = {"alltoall_info": object()}
    instances.update((first, second))
    comm = Mock()

    first.checkpoint_restore(comm)

    checkpoint_restore.assert_called_once_with(comm)
    assert first._dispatch_state == {}
    assert second._dispatch_state == {}


def test_two_sided_checkpoint_restore_noop_preserves_shared_owner_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = WeakSet()
    monkeypatch.setattr(NVLinkTwoSided, "_INSTANCES", instances)
    checkpoint_restore = Mock()
    monkeypatch.setattr(mnnvl.MnnvlMoe, "checkpoint_restore", checkpoint_restore)
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_workspace",
        Mock(mapped=True),
    )
    monkeypatch.setattr(
        mnnvl.MnnvlMoe,
        "moe_prepare_workspace",
        Mock(mapped=True),
    )
    first = NVLinkTwoSided.__new__(NVLinkTwoSided)
    first._dispatch_state = {"alltoall_info": object()}
    second = NVLinkTwoSided.__new__(NVLinkTwoSided)
    second._dispatch_state = {"alltoall_info": object()}
    instances.update((first, second))
    comm = Mock()

    first.checkpoint_restore(comm)

    checkpoint_restore.assert_called_once_with(comm)
    assert first._dispatch_state
    assert second._dispatch_state


@pytest.mark.parametrize("wrapper_type", [MoeAlltoAll, NVLinkOneSided])
def test_frontend_checkpoint_delegates_to_shared_lifecycle(
    wrapper_type: type[MoeAlltoAll] | type[NVLinkOneSided],
) -> None:
    wrapper = wrapper_type.__new__(wrapper_type)
    wrapper._workspace_lifecycle = Mock()
    comm = Mock()

    wrapper.checkpoint_prepare()
    wrapper.checkpoint_restore(comm)

    wrapper._workspace_lifecycle.checkpoint_prepare.assert_called_once_with()
    wrapper._workspace_lifecycle.checkpoint_restore.assert_called_once()
    assert wrapper._workspace_lifecycle.checkpoint_restore.call_args.args[0] is comm


@pytest.mark.parametrize("wrapper_type", [MoeAlltoAll, NVLinkOneSided])
def test_frontend_destroy_unregisters_from_shared_lifecycle(
    wrapper_type: type[MoeAlltoAll] | type[NVLinkOneSided],
) -> None:
    wrapper = wrapper_type.__new__(wrapper_type)
    wrapper._destroyed = False
    wrapper._workspace_registered = True
    lifecycle = Mock()
    wrapper._workspace_lifecycle = lifecycle
    if wrapper_type is NVLinkOneSided:
        wrapper._workspace_key = None

    wrapper.destroy()
    wrapper.destroy()

    lifecycle.unregister.assert_called_once_with(wrapper)


def test_moe_alltoall_aborted_registration_does_not_unregister(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = Mock()
    lifecycle.register.side_effect = RuntimeError("registration failed")
    monkeypatch.setattr(MoeAlltoAll, "_WORKSPACE", None)
    monkeypatch.setattr(MoeAlltoAll, "_init_constants", Mock())
    monkeypatch.setattr(
        MoeAlltoAll,
        "_METAINFO_INDEX",
        {
            "FLAG_VAL_OFFSET_INDEX": 0,
            "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX": 0,
            "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX": 0,
        },
    )
    monkeypatch.setattr(mnnvl.MnnvlMemory, "initialize", Mock())
    memory = Mock()
    memory.as_torch_strided_tensor.return_value = torch.zeros(1, dtype=torch.uint8)
    monkeypatch.setattr(
        "tensorrt_llm._torch.distributed.moe_alltoall.MnnvlMemory",
        Mock(return_value=memory),
    )
    monkeypatch.setattr(
        _MnnvlAlltoAllWorkspaceLifecycle,
        "get_or_create",
        Mock(return_value=lifecycle),
    )
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_a2a_initialize",
        Mock(return_value=torch.tensor([1])),
    )
    mapping = SimpleNamespace(moe_ep_size=2, moe_ep_rank=0)

    with pytest.raises(RuntimeError, match="registration failed"):
        MoeAlltoAll(
            mapping=mapping,
            max_num_tokens=1,
            top_k=1,
            num_slots=2,
            workspace_size_per_rank=1,
        )

    lifecycle.unregister.assert_not_called()


def test_one_sided_checkpoint_rejects_destroyed_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACES", {})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE_REFCOUNTS", {})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE", None)
    wrapper = NVLinkOneSided.__new__(NVLinkOneSided)
    wrapper._destroyed = False
    wrapper._workspace_lifecycle = Mock()
    wrapper._workspace_key = ("test",)
    wrapper._workspace_registered = True
    wrapper.destroy()

    with pytest.raises(RuntimeError, match="workspace has been destroyed"):
        wrapper.checkpoint_prepare()


def test_one_sided_finalizer_unregisters_from_shared_lifecycle() -> None:
    class _Lifecycle:
        def __init__(self) -> None:
            self.unregister_count = 0

        def unregister(self, client: object) -> None:
            self.unregister_count += 1

    lifecycle = _Lifecycle()
    wrapper = NVLinkOneSided.__new__(NVLinkOneSided)
    wrapper._destroyed = False
    wrapper._workspace_lifecycle = lifecycle
    wrapper._workspace_key = None
    wrapper._workspace_registered = True

    del wrapper
    gc.collect()

    assert lifecycle.unregister_count == 1


def test_one_sided_finalizer_preserves_collective_workspace_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_key = ("shared",)
    workspace_state = {"memory": object()}
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACES", {workspace_key: workspace_state})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE_REFCOUNTS", {workspace_key: 1})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE", workspace_state)
    wrapper = NVLinkOneSided.__new__(NVLinkOneSided)
    wrapper._workspace_lifecycle = Mock()
    wrapper._workspace_key = workspace_key
    wrapper._workspace_registered = True

    del wrapper
    gc.collect()

    assert NVLinkOneSided._WORKSPACES[workspace_key] is workspace_state
    assert NVLinkOneSided._WORKSPACE is workspace_state
    assert workspace_key not in NVLinkOneSided._WORKSPACE_REFCOUNTS


def test_one_sided_aborted_construction_does_not_release_sibling_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_key = ("shared",)
    workspace_state = {}
    refcounts = {workspace_key: 2}
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACES", {workspace_key: workspace_state})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE_REFCOUNTS", refcounts)
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE", workspace_state)
    wrapper = NVLinkOneSided.__new__(NVLinkOneSided)
    wrapper._destroyed = False
    wrapper._workspace_lifecycle = Mock()
    wrapper._workspace_key = workspace_key
    wrapper._workspace_registered = False

    wrapper.destroy()

    assert refcounts[workspace_key] == 2
    assert NVLinkOneSided._WORKSPACES[workspace_key] is workspace_state
    assert NVLinkOneSided._WORKSPACE is workspace_state


def test_one_sided_failed_registration_does_not_publish_new_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Memory:
        mapped = True

        @staticmethod
        def initialize() -> None:
            pass

        def __init__(self, mapping: object, size: int) -> None:
            self.comm = _FakeComm()

        def as_torch_strided_tensor(self, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(1, dtype=torch.uint8)

    lifecycle = Mock()
    lifecycle.register.side_effect = RuntimeError("registration failed")
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACES", {})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE_REFCOUNTS", {})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE", None)
    monkeypatch.setattr(NVLinkOneSided, "is_platform_supported", Mock(return_value=True))
    monkeypatch.setattr(NVLinkOneSided, "_init_constants", Mock())
    monkeypatch.setattr(NVLinkOneSided, "FLAG_VAL_OFFSET_INDEX", 0)
    monkeypatch.setattr(NVLinkOneSided, "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX", 0)
    monkeypatch.setattr(NVLinkOneSided, "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX", 0)
    monkeypatch.setattr(one_sided_module, "MnnvlMemory", _Memory)
    monkeypatch.setattr(
        _MnnvlAlltoAllWorkspaceLifecycle,
        "get_or_create",
        Mock(return_value=lifecycle),
    )
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_a2a_initialize",
        Mock(return_value=torch.tensor([1])),
    )
    mapping = SimpleNamespace(
        world_size=2,
        moe_ep_size=2,
        moe_ep_rank=0,
        has_cp_helix=Mock(return_value=False),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        NVLinkOneSided(
            mapping=mapping,
            num_slots=2,
            top_k=1,
            max_num_tokens_per_rank=1,
        )

    assert NVLinkOneSided._WORKSPACES == {}
    assert NVLinkOneSided._WORKSPACE_REFCOUNTS == {}
    assert NVLinkOneSided._WORKSPACE is None
    lifecycle.unregister.assert_not_called()
