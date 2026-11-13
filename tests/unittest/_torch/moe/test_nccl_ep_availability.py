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

import sys
from collections.abc import Iterator
from enum import IntEnum
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._utils as trtllm_utils
from tensorrt_llm._torch.moe.fused_moe import nccl_ep_utils as ep_utils
from tensorrt_llm._torch.moe.fused_moe.communication.nccl_ep import NcclEP

pytestmark = pytest.mark.cpu_only
_Layout = IntEnum("Layout", {"RANK_MAJOR": 0})
_GIN = SimpleNamespace(NONE=0, PROXY=2, GDAKI=3)


class _NativeError(Exception):
    pass


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    ep_utils.destroy_all_nccl_ep_contexts()
    comm = Mock(device_api_support=True, n_lsa_teams=1, gin_type=_GIN.NONE)
    core = SimpleNamespace(
        get_unique_id=Mock(return_value=b"unique-id"),
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
    monkeypatch.setitem(sys.modules, "nccl.core.typing", SimpleNamespace(NcclGinType=_GIN))
    monkeypatch.setitem(sys.modules, "nccl.ep", ep)
    mpi = Mock()
    mpi.Get_rank.return_value, mpi.Get_size.return_value = 0, 2
    mpi.bcast.side_effect = lambda value, root: value
    mpi.allreduce.side_effect = mpi.allgather.side_effect = AssertionError("extra collective")
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
    yield SimpleNamespace(comm=comm, core=core, ep=ep, mpi=mpi, session=session, mapping=mapping)
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
    runtime.mpi.bcast.assert_called_once_with(b"unique-id", root=0)
    runtime.ep.Group.create.assert_not_called()
    runtime.session.Split.assert_not_called()
    runtime.session.Get_group.return_value.Incl.assert_called_once_with([4, 7])
    if not supported:
        runtime.comm.destroy.assert_called_once()
        runtime.mpi.Free.assert_called_once()
    assert runtime.mpi.allreduce.call_count == runtime.mpi.allgather.call_count == 0


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


@pytest.mark.parametrize("failure", ["group", "allocation", "allocation_and_cleanup"])
def test_context_initialization_failure_releases_transferred_resources(
    runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    failure: Literal["group", "allocation", "allocation_and_cleanup"],
) -> None:
    strategy = NcclEP(runtime.mapping, 4, 512, max_num_tokens=8, top_k=2)
    error = _NativeError("initialization failed")
    group = runtime.ep.Group.create.return_value
    if failure == "group":
        runtime.ep.Group.create.side_effect = error
    else:
        empty = torch.empty
        monkeypatch.setattr(torch, "empty", Mock(side_effect=[empty(2), error]))
        if failure == "allocation_and_cleanup":
            group.destroy.side_effect = _NativeError("cleanup failed")
    contexts = []
    destroy = ep_utils.NcclEpContext.destroy

    def record_destroy(context: ep_utils.NcclEpContext) -> None:
        contexts.append(context)
        destroy(context)

    monkeypatch.setattr(ep_utils.NcclEpContext, "destroy", record_destroy)
    with pytest.raises(_NativeError) as raised:
        strategy._get_context()
    assert raised.value is error
    assert strategy._ctx is None
    (context,) = contexts
    assert context.comm is context._ep_mpi_comm is context.ep_group is None
    assert context.output_tokens_buf is context.recv_topk_idx_buf is None
    assert not ep_utils._ep_group_cache
    strategy.destroy()
    context.destroy()
    ep_utils.destroy_all_nccl_ep_contexts()
    runtime.comm.finalize.assert_called_once()
    runtime.comm.destroy.assert_called_once()
    runtime.mpi.Free.assert_called_once()
    if failure != "group":
        group.destroy.assert_called_once()
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
    with pytest.raises(ValueError, match="initialization/capability query failed"):
        ep_utils.is_nccl_ep_installed(runtime.mapping)
    runtime.mpi.Free.assert_called_once()
    runtime.core.Communicator.init.side_effect = None
    runtime.comm.device_api_support = True
    assert ep_utils.is_nccl_ep_installed(runtime.mapping)
    assert runtime.core.Communicator.init.call_count == 2


def test_mpi_unique_id_failure_uses_the_existing_bootstrap_broadcast(
    runtime: SimpleNamespace,
) -> None:
    runtime.core.get_unique_id.side_effect = _NativeError("unique ID failed")
    with pytest.raises(ValueError, match="unique ID"):
        ep_utils.is_nccl_ep_installed(runtime.mapping)
    runtime.mpi.bcast.assert_called_once_with(None, root=0)
    runtime.core.Communicator.init.assert_not_called()
    runtime.mpi.Free.assert_called_once()


@pytest.mark.parametrize("fails", [False, True])
def test_torch_path_reuses_interop_without_capability_collectives(
    runtime: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: True)
    for name in ("broadcast", "all_reduce", "all_gather_object"):
        monkeypatch.setattr(dist, name, Mock(side_effect=AssertionError("extra collective")))
    create = Mock(
        return_value=runtime.comm, side_effect=_NativeError("interop failed") if fails else None
    )
    monkeypatch.setitem(
        sys.modules, "nccl.ep.interop.torch", SimpleNamespace(get_nccl_comm_from_group=create)
    )
    if fails:
        with pytest.raises(ValueError, match="interop failed"):
            ep_utils.is_nccl_ep_installed(runtime.mapping)
    else:
        assert ep_utils.is_nccl_ep_installed(runtime.mapping)
        assert ep_utils._take_ep_nccl_comm(runtime.mapping) == (runtime.comm, None)
        ep_utils._release_ep_nccl_comm(runtime.comm, None)
    create.assert_called_once_with(runtime.mapping.moe_ep_group_pg)
    runtime.core.Communicator.init.assert_not_called()
