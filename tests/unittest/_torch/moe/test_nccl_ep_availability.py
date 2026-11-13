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

"""Availability checks retain the real communicator for the lazy EP context."""

from __future__ import annotations

import multiprocessing
import sys
from collections.abc import Iterator
from datetime import timedelta
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._utils as trtllm_utils
from tensorrt_llm._errors import RuntimeInitError
from tensorrt_llm._torch.moe.fused_moe import nccl_ep_utils as ep_utils
from tensorrt_llm._torch.moe.fused_moe.communication.nccl_ep import NcclEP

pytestmark = pytest.mark.cpu_only
_Layout = IntEnum("Layout", {"RANK_MAJOR": 0})
_GIN = SimpleNamespace(NONE=0, PROXY=2, GDAKI=3)


class _NativeError(Exception):
    pass


class _UniqueId:
    def __init__(self, data: bytes = bytes(128)) -> None:
        self.as_bytes = data

    @classmethod
    def from_bytes(cls, data: bytes) -> "_UniqueId":
        return cls(data)


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    ep_utils.destroy_all_nccl_ep_contexts()
    comm = Mock(device_api_support=True, n_lsa_teams=1, gin_type=_GIN.NONE)
    comm.ptr = comm
    bindings = SimpleNamespace(
        comm_query_properties=Mock(
            side_effect=lambda ptr: SimpleNamespace(
                device_api_support=ptr.device_api_support,
                n_lsa_teams=ptr.n_lsa_teams,
                gin_type=ptr.gin_type,
            )
        )
    )
    core = SimpleNamespace(
        UniqueId=_UniqueId,
        get_unique_id=Mock(return_value=_UniqueId(b"x" * 128)),
        Communicator=SimpleNamespace(init=Mock(return_value=comm)),
    )
    ep = SimpleNamespace(
        Layout=_Layout,
        GroupConfig=SimpleNamespace,
        Group=SimpleNamespace(create=Mock(return_value=Mock())),
        DispatchConfig=SimpleNamespace,
        DispatchOutputs=SimpleNamespace,
        DispatchQuantizationRecipe=SimpleNamespace(NONE=0),
        Algorithm=SimpleNamespace(LOW_LATENCY=0),
        LayoutInfo=SimpleNamespace,
        Tensor=lambda tensor: tensor,
        get_lib_version=Mock(return_value="0.2.0"),
    )
    monkeypatch.setitem(sys.modules, "nccl", SimpleNamespace(core=core, ep=ep))
    monkeypatch.setitem(sys.modules, "nccl.core", core)
    monkeypatch.setitem(sys.modules, "nccl.bindings", SimpleNamespace(nccl=bindings))
    monkeypatch.setitem(sys.modules, "nccl.core.typing", SimpleNamespace(NcclGinType=_GIN))
    monkeypatch.setitem(sys.modules, "nccl.ep", ep)
    mpi = Mock()
    mpi.Get_rank.return_value, mpi.Get_size.return_value = 0, 2
    mpi.bcast.side_effect = lambda value, root: value
    mpi.allreduce.side_effect = mpi.allgather.side_effect = AssertionError(
        "unexpected capability vote"
    )
    session = Mock()
    session.Create_group.return_value = mpi
    monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: False)
    monkeypatch.setattr(trtllm_utils, "mpi_comm", lambda: session)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(ep_utils, "_nccl_ep_installed", None)
    monkeypatch.setattr(ep_utils, "register_moe_comm_timeout_proxy", Mock())
    monkeypatch.setattr(ep_utils, "unregister_moe_comm_timeout_proxy", Mock())
    mapping = SimpleNamespace(
        moe_ep_group=[4, 7], moe_ep_group_pg=object(), moe_ep_size=2, moe_ep_rank=0
    )
    yield SimpleNamespace(
        comm=comm, core=core, bindings=bindings, ep=ep, mpi=mpi, session=session, mapping=mapping
    )
    ep_utils.destroy_all_nccl_ep_contexts()


def test_module_only_check_does_not_initialize_cuda_or_nccl(
    runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("CUDA init")))
    assert ep_utils.is_nccl_ep_installed() and ep_utils.is_nccl_ep_installed()
    runtime.ep.get_lib_version.assert_called_once()
    runtime.core.Communicator.init.assert_not_called()


@pytest.mark.parametrize(
    ("device_api", "teams", "gin", "supported"),
    [
        (True, 1, _GIN.NONE, True),
        (False, 1, _GIN.NONE, False),
        (True, 2, _GIN.NONE, False),
        (True, 2, _GIN.PROXY, True),
        (True, 2, _GIN.GDAKI, True),
        (True, 2, 5, True),
    ],
)
def test_capabilities_and_cached_decision(
    runtime: SimpleNamespace, device_api: bool, teams: int, gin: int, supported: bool
) -> None:
    runtime.comm.device_api_support, runtime.comm.n_lsa_teams, runtime.comm.gin_type = (
        device_api,
        teams,
        gin,
    )
    assert ep_utils.is_nccl_ep_installed(runtime.mapping) is supported
    assert ep_utils.is_nccl_ep_installed(runtime.mapping) is supported
    runtime.core.Communicator.init.assert_called_once()
    runtime.mpi.bcast.assert_called_once_with(runtime.core.get_unique_id.return_value, root=0)
    runtime.ep.Group.create.assert_not_called()
    runtime.session.Split.assert_not_called()
    runtime.session.Get_group.return_value.Incl.assert_called_once_with([4, 7])
    if not supported:
        runtime.comm.destroy.assert_called_once()
        runtime.mpi.Free.assert_called_once()
    runtime.mpi.allreduce.assert_not_called()
    runtime.mpi.allgather.assert_not_called()


def test_lazy_context_consumes_checked_comm_and_preserves_refcount(
    runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = torch.empty
    allocate = Mock(
        side_effect=lambda *args, **kwargs: empty(*args, **(kwargs | {"device": "cpu"}))
    )
    monkeypatch.setattr(torch, "empty", allocate)
    first = NcclEP(runtime.mapping, 4, 512, max_num_tokens=8, top_k=2)
    second = NcclEP(runtime.mapping, 4, 512, max_num_tokens=8, top_k=2)
    allocate.assert_not_called()
    assert first._ctx is second._ctx is None
    assert first._get_context() is second._get_context()
    runtime.ep.Group.create.assert_called_once()
    assert runtime.ep.Group.create.call_args.args[0] is runtime.comm
    runtime.core.Communicator.init.assert_called_once()
    first.destroy()
    runtime.comm.destroy.assert_not_called()
    second.destroy()
    runtime.comm.finalize.assert_called_once()
    runtime.comm.destroy.assert_called_once()


@pytest.mark.parametrize("finalize_fails", [False, True])
def test_unused_comm_cleanup(runtime: SimpleNamespace, finalize_fails: bool) -> None:
    strategy = NcclEP(runtime.mapping, 4, 512, max_num_tokens=8, top_k=2)
    if finalize_fails:
        runtime.comm.finalize.side_effect = _NativeError("finalize failed")
    strategy.destroy()
    runtime.comm.finalize.assert_called_once()
    runtime.comm.destroy.assert_called_once()
    runtime.mpi.Free.assert_called_once()


@pytest.mark.parametrize(
    "failure", ["group", "allocation", "allocation_and_abort", "allocation_and_unregister"]
)
def test_context_initialization_failure_aborts_without_collective_cleanup(
    runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal["group", "allocation", "allocation_and_abort", "allocation_and_unregister"],
) -> None:
    strategy = NcclEP(runtime.mapping, 4, 512, max_num_tokens=8, top_k=2)
    error = (
        _NativeError("group creation failed")
        if failure == "group"
        else torch.OutOfMemoryError("receive-buffer allocation failed")
    )
    group = runtime.ep.Group.create.return_value
    for release in (group.destroy, runtime.comm.finalize, runtime.comm.destroy, runtime.mpi.Free):
        release.side_effect = AssertionError("collective cleanup on initialization failure")
    if failure == "group":
        runtime.ep.Group.create.side_effect = error
    else:
        empty = torch.empty
        monkeypatch.setattr(torch, "empty", Mock(side_effect=[empty(2), error]))
        if failure == "allocation_and_abort":
            runtime.comm.abort.side_effect = _NativeError("abort failed")
        elif failure == "allocation_and_unregister":
            ep_utils.unregister_moe_comm_timeout_proxy.side_effect = _NativeError(
                "unregister failed"
            )
    contexts = []
    abort = ep_utils.NcclEpContext._abort_initialization

    def record_abort(context: ep_utils.NcclEpContext) -> None:
        contexts.append(context)
        abort(context)

    monkeypatch.setattr(ep_utils.NcclEpContext, "_abort_initialization", record_abort)
    with pytest.raises(type(error)) as raised:
        strategy._get_context()
    assert raised.value is error
    assert strategy._ctx is None
    (context,) = contexts
    assert (
        context.comm is context._ep_mpi_comm is context.ep_group is context._timeout_proxy is None
    )
    assert context.output_tokens_buf is context.recv_topk_idx_buf is None
    assert not ep_utils._ep_group_cache
    strategy.destroy()
    context.destroy()
    ep_utils.destroy_all_nccl_ep_contexts()
    runtime.comm.abort.assert_called_once()
    for release in (group.destroy, runtime.comm.finalize, runtime.comm.destroy, runtime.mpi.Free):
        release.assert_not_called()
    if failure != "group":
        ep_utils.unregister_moe_comm_timeout_proxy.assert_called_once()


@pytest.mark.parametrize("failure", ["alignment", "version"])
def test_local_validation_precedes_communicator_creation(
    runtime: SimpleNamespace, failure: Literal["alignment", "version"]
) -> None:
    quant = SimpleNamespace(
        layer_quant_mode=SimpleNamespace(has_fp8_qdq=lambda: False, has_nvfp4=lambda: True)
    )
    if failure == "version":
        runtime.ep.get_lib_version.return_value = "0.1.0"
    with pytest.raises(RuntimeError, match="divisible by 256|requires libnccl_ep"):
        NcclEP(runtime.mapping, 4, 128 if failure == "alignment" else 512, quant_config=quant)
    runtime.core.Communicator.init.assert_not_called()
    runtime.session.Create_group.assert_not_called()
    assert not ep_utils._ep_comm_cache


def test_pending_cleanup_uses_owning_device(
    runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    other_comm = Mock(device_api_support=True, n_lsa_teams=1, gin_type=_GIN.NONE)
    other_comm.ptr = other_comm
    other_mpi = Mock()
    other_mpi.Get_rank.return_value, other_mpi.Get_size.return_value = 0, 2
    other_mpi.bcast.side_effect = lambda value, root: value
    runtime.core.Communicator.init.side_effect = [runtime.comm, other_comm]
    runtime.session.Create_group.side_effect = [runtime.mpi, other_mpi]
    first = NcclEP(runtime.mapping, 4, 512)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    second = NcclEP(runtime.mapping, 4, 512)
    first.destroy()
    runtime.comm.destroy.assert_called_once()
    runtime.mpi.Free.assert_called_once()
    other_comm.destroy.assert_not_called()
    other_mpi.Free.assert_not_called()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    second.destroy()
    first.destroy()
    ep_utils.destroy_all_nccl_ep_contexts()
    for comm, mpi in ((runtime.comm, runtime.mpi), (other_comm, other_mpi)):
        comm.finalize.assert_called_once()
        comm.destroy.assert_called_once()
        mpi.Free.assert_called_once()


def test_unsupported_gate_and_cache_reset(runtime: SimpleNamespace) -> None:
    runtime.comm.device_api_support = False
    with pytest.raises(RuntimeError, match="Device API/GIN"):
        NcclEP(runtime.mapping, 4, 512)
    runtime.comm.device_api_support = True
    assert not ep_utils.is_nccl_ep_installed(runtime.mapping)
    ep_utils.destroy_all_nccl_ep_contexts()
    assert ep_utils.is_nccl_ep_installed(runtime.mapping)
    assert runtime.core.Communicator.init.call_count == 2


@pytest.mark.parametrize("failure", ["init", "query"])
def test_errors_are_not_cached_as_unsupported(
    runtime: SimpleNamespace, failure: Literal["init", "query"]
) -> None:
    if failure == "init":
        runtime.core.Communicator.init.side_effect = _NativeError("init failed")
    else:
        del runtime.comm.device_api_support
    with pytest.raises(RuntimeInitError, match="initialization/capability query failed"):
        ep_utils.is_nccl_ep_installed(runtime.mapping)
    runtime.comm.finalize.assert_not_called()
    runtime.comm.destroy.assert_not_called()
    runtime.mpi.Free.assert_not_called()
    assert runtime.comm.abort.call_count == (0 if failure == "init" else 1)
    runtime.core.Communicator.init.side_effect = None
    runtime.comm.device_api_support = True
    assert ep_utils.is_nccl_ep_installed(runtime.mapping)
    assert runtime.core.Communicator.init.call_count == 2


def test_mpi_unique_id_failure_uses_the_existing_bootstrap_broadcast(
    runtime: SimpleNamespace,
) -> None:
    runtime.core.get_unique_id.side_effect = _NativeError("unique ID failed")
    with pytest.raises(RuntimeInitError, match="unique ID"):
        ep_utils.is_nccl_ep_installed(runtime.mapping)
    runtime.mpi.bcast.assert_called_once_with(None, root=0)
    runtime.core.Communicator.init.assert_not_called()
    runtime.comm.abort.assert_not_called()
    runtime.mpi.Free.assert_not_called()


def test_torch_bootstrap_uses_ep_group_without_capability_collectives(
    runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda _: 0)
    monkeypatch.setattr(dist, "get_backend", lambda _: "gloo")
    monkeypatch.setattr(dist, "get_process_group_ranks", lambda _: [4, 7])
    broadcast, reduce = Mock(), Mock()
    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "all_reduce", reduce)
    assert ep_utils.is_nccl_ep_installed(runtime.mapping)
    assert ep_utils._take_ep_nccl_comm(runtime.mapping) == (runtime.comm, None)
    ep_utils._release_ep_nccl_comm(runtime.comm, None)
    broadcast.assert_called_once()
    assert broadcast.call_args.kwargs == {"src": 4, "group": runtime.mapping.moe_ep_group_pg}
    reduce.assert_not_called()
    runtime.session.Create_group.assert_not_called()


def test_raw_gin_property_accepts_transport_unknown_to_old_enum(
    runtime: SimpleNamespace,
) -> None:
    del runtime.comm.gin_type
    runtime.bindings.comm_query_properties.side_effect = None
    runtime.bindings.comm_query_properties.return_value = SimpleNamespace(
        device_api_support=True, n_lsa_teams=2, gin_type=5
    )
    assert ep_utils.is_nccl_ep_installed(runtime.mapping)


def _check_ep_bootstrap(rank: int, rendezvous: str, scenario: str) -> None:
    """Run the actual factory and EP control collectives with fake native endpoints."""
    import torch.distributed as dist

    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode
    from tensorrt_llm._torch.moe.fused_moe.communication import communication_factory

    patch = pytest.MonkeyPatch()
    fixture = runtime.__wrapped__(patch)
    state = next(fixture)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=5),
    )
    try:
        patch.setattr(trtllm_utils, "mpi_disabled", lambda: True)
        state.mapping.moe_ep_group, state.mapping.moe_ep_group_pg = [0, 1], dist.group.WORLD
        state.mapping.moe_ep_rank = rank
        state.mapping.moe_tp_size, state.mapping.dp_size = 1, 2
        state.mapping.enable_attention_dp, state.mapping.has_cp_helix = True, lambda: False
        factory = communication_factory.CommunicationFactory
        patch.setattr(factory, "_get_nccl_ep_unavailable_reason", lambda *args: None)
        for backend in ("NVLinkOneSided", "NVLinkTwoSided", "DeepEP", "DeepEPLowLatency"):
            patch.setattr(
                communication_factory, backend, Mock(side_effect=RuntimeError("unavailable"))
            )
        config = SimpleNamespace(
            mapping=state.mapping,
            pretrained_config=SimpleNamespace(hidden_size=512),
            torch_dtype=torch.bfloat16,
            quant_config=None,
            max_num_tokens=8,
            moe_max_num_tokens=None,
            use_cuda_graph=False,
            use_low_precision_moe_combine=False,
            moe_load_balancer=None,
        )
        if scenario == "root_id_failure" and rank == 0:
            state.core.get_unique_id.side_effect = _NativeError("unique ID failed")
        elif scenario == "unsupported":
            state.comm.device_api_support = False
        elif scenario == "query_failure":
            state.bindings.comm_query_properties.side_effect = _NativeError("query failed")
        elif scenario == "init_failure":
            state.core.Communicator.init.side_effect = _NativeError("init failed")
        elif scenario == "unknown_gin":
            state.comm.n_lsa_teams, state.comm.gin_type = 2, 5
        broadcast, reduce = Mock(wraps=dist.broadcast), Mock(wraps=dist.all_reduce)
        patch.setattr(dist, "broadcast", broadcast)
        patch.setattr(dist, "all_reduce", reduce)
        with MetaInitMode():
            if scenario in ("root_id_failure", "query_failure", "init_failure"):
                with pytest.raises(RuntimeInitError):
                    factory.create_strategy(config, 4, 4, 2, 2)
            else:
                strategy = factory.create_strategy(config, 4, 4, 2, 2)
                expected = (
                    communication_factory.AllGatherReduceScatter
                    if scenario == "unsupported"
                    else NcclEP
                )
                assert isinstance(strategy, expected)
                assert ep_utils.is_nccl_ep_installed(state.mapping) is (scenario == "unknown_gin")
                strategy.destroy()
        broadcast.assert_called_once()
        assert broadcast.call_args.kwargs == {"src": 0, "group": dist.group.WORLD}
        reduce.assert_not_called()
        if scenario == "root_id_failure":
            state.core.Communicator.init.assert_not_called()
        elif scenario in ("init_failure", "query_failure"):
            state.comm.finalize.assert_not_called()
            state.comm.destroy.assert_not_called()
            assert state.comm.abort.call_count == (1 if scenario == "query_failure" else 0)
        else:
            state.comm.finalize.assert_called_once()
            state.comm.destroy.assert_called_once()
            state.comm.abort.assert_not_called()
        state.session.Create_group.assert_not_called()
    finally:
        dist.destroy_process_group()
        next(fixture, None)
        patch.undo()


@pytest.mark.parametrize(
    "scenario", ["root_id_failure", "unsupported", "query_failure", "init_failure", "unknown_gin"]
)
def test_two_rank_ep_bootstrap(tmp_path: Path, scenario: str) -> None:
    """Uniform EP configurations bootstrap under MetaInitMode without a capability vote."""
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(
            target=_check_ep_bootstrap, args=(rank, (tmp_path / "store").as_uri(), scenario)
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(15)
        assert all(process.exitcode == 0 for process in processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
