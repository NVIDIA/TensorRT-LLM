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
"""CPU-only tests for the NCCL-EP context bootstrap in ``nccl_ep_utils``.

``NcclEpContext`` seeds its NCCL communicator from the torch EP process group
under the Ray orchestrator (``mpi_disabled()``) and from an ``MPI_COMM_WORLD``
split keyed by pipeline stage otherwise. These tests stub the ``nccl`` namespace
package, ``mpi4py`` and the CUDA allocations, so they run in the CPU-only stage
next to ``test_communication_factory.py`` and catch a regression that selects the
wrong bootstrap, passes the wrong EP group, or tears the resources down in the
wrong order.
"""

from __future__ import annotations

import enum
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from tensorrt_llm import _utils as trtllm_utils
from tensorrt_llm._torch.moe.fused_moe import nccl_ep_utils

_NUM_EXPERTS = 8
_MAX_TOKENS_PER_RANK = 4
_HIDDEN_SIZE = 512
_MAX_TOP_K = 2
_UNIQUE_ID = b"nccl-unique-id"


class _Algorithm(enum.Enum):
    LOW_LATENCY = 1


class _Layout(enum.IntEnum):
    RANK_MAJOR = 0
    EXPERT_MAJOR = 1


class _ExpertIdKind(enum.Enum):
    LOCAL = 0
    GLOBAL = 1


class _DispatchQuantizationRecipe(enum.Enum):
    NONE = 0
    FWD = 1
    DS_FP8E3M4 = 2


class _GroupConfig:
    """Plain stand-in without ``__dataclass_fields__``: the zero-copy probe reads False."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _LayoutInfo:
    """Stand-in without ``recv_topk_idx_kind``: the global-id probe reads False."""

    def __init__(self, src_rank_counters=None) -> None:
        self.src_rank_counters = src_rank_counters
        self._lowpp = SimpleNamespace()


class _Tensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _CpuTorch:
    """Forward to ``torch`` but keep the context's receive buffers on the CPU."""

    cuda = SimpleNamespace(
        current_device=lambda: 0,
        current_stream=lambda: SimpleNamespace(cuda_stream=0),
    )

    @staticmethod
    def device(*args, **kwargs) -> torch.device:
        return torch.device("cpu")

    def __getattr__(self, name: str):
        return getattr(torch, name)


def _make_mapping() -> SimpleNamespace:
    return SimpleNamespace(
        moe_ep_size=2,
        moe_ep_rank=1,
        pp_rank=3,
        moe_ep_group_pg=object(),
    )


def _install_fake_nccl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mpi_comm_result: Mock,
    torch_group_comm: Mock,
    ep_group: Mock,
) -> SimpleNamespace:
    """Install stub ``nccl.core`` / ``nccl.ep`` / ``nccl.ep.interop.torch`` modules."""
    nccl = ModuleType("nccl")
    nccl.__path__ = []
    core = ModuleType("nccl.core")
    core.get_unique_id = Mock(return_value=_UNIQUE_ID)
    core.Communicator = SimpleNamespace(init=Mock(return_value=mpi_comm_result))
    ep = ModuleType("nccl.ep")
    ep.__path__ = []
    ep.Algorithm = _Algorithm
    ep.Layout = _Layout
    ep.ExpertIdKind = _ExpertIdKind
    ep.DispatchQuantizationRecipe = _DispatchQuantizationRecipe
    ep.GroupConfig = _GroupConfig
    ep.Group = SimpleNamespace(create=Mock(return_value=ep_group))
    ep.LayoutInfo = _LayoutInfo
    ep.Tensor = _Tensor
    ep.DispatchConfig = Mock(name="DispatchConfig")
    ep.DispatchOutputs = Mock(name="DispatchOutputs")
    ep.get_lib_version = Mock(return_value="0.2.0")
    interop = ModuleType("nccl.ep.interop")
    interop.__path__ = []
    interop_torch = ModuleType("nccl.ep.interop.torch")
    interop_torch.get_nccl_comm_from_group = Mock(return_value=torch_group_comm)
    nccl.core = core
    nccl.ep = ep
    ep.interop = interop
    interop.torch = interop_torch
    for name, module in (
        ("nccl", nccl),
        ("nccl.core", core),
        ("nccl.ep", ep),
        ("nccl.ep.interop", interop),
        ("nccl.ep.interop.torch", interop_torch),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # destroy() imports mpi4py only on the MPI branch; keep the test independent of it.
    monkeypatch.setitem(
        sys.modules, "mpi4py", SimpleNamespace(MPI=SimpleNamespace(Exception=Exception))
    )
    return SimpleNamespace(core=core, ep=ep, interop_torch=interop_torch)


@pytest.fixture
def bootstrap(monkeypatch: pytest.MonkeyPatch):
    """Patch everything around ``NcclEpContext.__init__`` except the bootstrap branch."""
    torch_group_comm = Mock(name="torch_group_comm")
    mpi_comm_result = Mock(name="mpi_seeded_comm")
    ep_group = Mock(name="ep_group")
    fake = _install_fake_nccl(
        monkeypatch,
        mpi_comm_result=mpi_comm_result,
        torch_group_comm=torch_group_comm,
        ep_group=ep_group,
    )
    sub_comm = Mock(name="ep_mpi_sub_comm")
    sub_comm.Get_size.return_value = 2
    sub_comm.bcast.return_value = _UNIQUE_ID
    mpi_comm = Mock(name="mpi_comm")
    mpi_comm.return_value.Split.return_value = sub_comm
    monkeypatch.setattr(trtllm_utils, "mpi_comm", mpi_comm)
    monkeypatch.setattr(nccl_ep_utils, "nccl_ep_supports_version", lambda _minimum: True)
    monkeypatch.setattr(nccl_ep_utils, "torch", _CpuTorch())
    register = Mock(name="register_moe_comm_timeout_proxy")
    unregister = Mock(name="unregister_moe_comm_timeout_proxy")
    monkeypatch.setattr(nccl_ep_utils, "register_moe_comm_timeout_proxy", register)
    monkeypatch.setattr(nccl_ep_utils, "unregister_moe_comm_timeout_proxy", unregister)
    return SimpleNamespace(
        fake=fake,
        torch_group_comm=torch_group_comm,
        mpi_comm_result=mpi_comm_result,
        ep_group=ep_group,
        sub_comm=sub_comm,
        mpi_comm=mpi_comm,
        register=register,
        unregister=unregister,
        monkeypatch=monkeypatch,
    )


def _create_context(mapping: SimpleNamespace) -> nccl_ep_utils.NcclEpContext:
    return nccl_ep_utils.NcclEpContext(
        mapping,
        num_experts=_NUM_EXPERTS,
        max_tokens_per_rank=_MAX_TOKENS_PER_RANK,
        hidden_size=_HIDDEN_SIZE,
        max_top_k=_MAX_TOP_K,
    )


def test_ray_orchestrator_seeds_comm_from_torch_ep_group(bootstrap) -> None:
    bootstrap.monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: True)
    mapping = _make_mapping()

    ctx = _create_context(mapping)

    bootstrap.fake.interop_torch.get_nccl_comm_from_group.assert_called_once_with(
        mapping.moe_ep_group_pg
    )
    bootstrap.mpi_comm.assert_not_called()
    bootstrap.fake.core.get_unique_id.assert_not_called()
    bootstrap.fake.core.Communicator.init.assert_not_called()
    assert ctx._ep_mpi_comm is None
    assert ctx.comm is bootstrap.torch_group_comm
    assert ctx.ep_group is bootstrap.ep_group

    # The group is created from the torch-seeded comm with the LL rank-major budget.
    create = bootstrap.fake.ep.Group.create
    create.assert_called_once()
    comm_arg, cfg = create.call_args.args
    assert comm_arg is bootstrap.torch_group_comm
    assert cfg.kwargs == {
        "algorithm": _Algorithm.LOW_LATENCY,
        "num_experts": _NUM_EXPERTS,
        "max_dispatch_tokens_per_rank": _MAX_TOKENS_PER_RANK,
        "max_recv_tokens_per_rank": mapping.moe_ep_size * _MAX_TOKENS_PER_RANK,
        "max_token_bytes": _HIDDEN_SIZE * 2,
    }
    bootstrap.register.assert_called_once()
    assert bootstrap.register.call_args.args[0]._group is bootstrap.ep_group


@pytest.mark.parametrize("ep_world_rank", [0, 1])
def test_mpi_orchestrator_splits_comm_world_by_pipeline_stage(bootstrap, ep_world_rank) -> None:
    bootstrap.monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: False)
    bootstrap.sub_comm.Get_rank.return_value = ep_world_rank
    mapping = _make_mapping()

    ctx = _create_context(mapping)

    bootstrap.mpi_comm.return_value.Split.assert_called_once_with(
        mapping.pp_rank, mapping.moe_ep_rank
    )
    bootstrap.fake.interop_torch.get_nccl_comm_from_group.assert_not_called()
    if ep_world_rank == 0:
        bootstrap.fake.core.get_unique_id.assert_called_once_with()
        bootstrap.sub_comm.bcast.assert_called_once_with(_UNIQUE_ID, root=0)
    else:
        bootstrap.fake.core.get_unique_id.assert_not_called()
        bootstrap.sub_comm.bcast.assert_called_once_with(None, root=0)
    bootstrap.fake.core.Communicator.init.assert_called_once_with(
        nranks=2, rank=ep_world_rank, unique_id=_UNIQUE_ID
    )
    assert ctx._ep_mpi_comm is bootstrap.sub_comm
    assert ctx.comm is bootstrap.mpi_comm_result
    assert bootstrap.fake.ep.Group.create.call_args.args[0] is bootstrap.mpi_comm_result


@pytest.mark.parametrize("mpi_disabled", [True, False])
def test_destroy_releases_group_then_comm_and_frees_mpi_only_when_split(
    bootstrap, mpi_disabled
) -> None:
    bootstrap.monkeypatch.setattr(trtllm_utils, "mpi_disabled", lambda: mpi_disabled)
    ctx = _create_context(_make_mapping())
    comm = ctx.comm
    proxy = bootstrap.register.call_args.args[0]
    order = Mock(name="teardown_order")
    order.attach_mock(bootstrap.unregister, "unregister")
    order.attach_mock(bootstrap.ep_group.destroy, "group_destroy")
    order.attach_mock(comm.finalize, "comm_finalize")
    order.attach_mock(comm.destroy, "comm_destroy")
    order.attach_mock(bootstrap.sub_comm.Free, "mpi_free")

    ctx.destroy()

    bootstrap.unregister.assert_called_once_with(proxy)
    # The timeout proxy must be unregistered before the group it points at is destroyed.
    expected = [
        call.unregister(proxy),
        call.group_destroy(),
        call.comm_finalize(),
        call.comm_destroy(),
    ]
    if not mpi_disabled:
        expected.append(call.mpi_free())
    assert order.mock_calls == expected
    assert ctx.ep_group is None
    assert ctx.comm is None
    assert ctx._ep_mpi_comm is None
    assert ctx.output_tokens_buf is None
