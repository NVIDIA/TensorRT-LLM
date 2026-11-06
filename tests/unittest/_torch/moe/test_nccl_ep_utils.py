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
under the Ray orchestrator (``mpi_disabled()``) and from a split of the session
MPI comm (``mpi_comm()``) keyed by pipeline stage otherwise. These tests stub
the ``nccl`` namespace package and the CUDA allocations, so they run in the
CPU-only stage and catch a regression that selects the wrong bootstrap, passes
the wrong EP group, or tears the resources down in the wrong order.
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

# The CPU stage runs -m cpu_only and collects only files that carry this marker.
pytestmark = pytest.mark.cpu_only

_EP_SIZE = 2
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

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _LayoutInfo:
    """Stand-in without ``recv_topk_idx_kind``: the global-id probe reads False."""

    def __init__(self, src_rank_counters: _Tensor | None = None) -> None:
        self.src_rank_counters = src_rank_counters
        self._lowpp = SimpleNamespace()


class _Tensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor


class _CpuTorch:
    """Forward to ``torch`` but keep the context's receive buffers on the CPU."""

    cuda = SimpleNamespace(current_device=lambda: 0)

    @staticmethod
    def device(*args: object, **kwargs: object) -> torch.device:
        return torch.device("cpu")

    def __getattr__(self, name: str) -> object:
        return getattr(torch, name)


def _make_mapping() -> SimpleNamespace:
    return SimpleNamespace(
        moe_ep_size=_EP_SIZE,
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
    return SimpleNamespace(core=core, ep=ep, interop_torch=interop_torch)


@pytest.fixture
def nccl_ep_fakes(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch everything around ``NcclEpContext.__init__`` except the bootstrap branch.

    Tests pick the branch with ``TLLM_DISABLE_MPI``, so the real ``mpi_disabled()`` runs.
    """
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
    sub_comm.Get_size.return_value = _EP_SIZE
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
    )


def _create_context(mapping: SimpleNamespace) -> nccl_ep_utils.NcclEpContext:
    return nccl_ep_utils.NcclEpContext(
        mapping,
        num_experts=_NUM_EXPERTS,
        max_tokens_per_rank=_MAX_TOKENS_PER_RANK,
        hidden_size=_HIDDEN_SIZE,
        max_top_k=_MAX_TOP_K,
    )


def test_ray_orchestrator_seeds_comm_from_torch_ep_group(
    nccl_ep_fakes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    mapping = _make_mapping()

    ctx = _create_context(mapping)

    nccl_ep_fakes.fake.interop_torch.get_nccl_comm_from_group.assert_called_once_with(
        mapping.moe_ep_group_pg
    )
    nccl_ep_fakes.mpi_comm.assert_not_called()
    nccl_ep_fakes.fake.core.get_unique_id.assert_not_called()
    nccl_ep_fakes.fake.core.Communicator.init.assert_not_called()
    assert ctx._ep_mpi_comm is None
    assert ctx.comm is nccl_ep_fakes.torch_group_comm
    assert ctx.ep_group is nccl_ep_fakes.ep_group

    # The group is created from the torch-seeded comm with the LL rank-major budget.
    create = nccl_ep_fakes.fake.ep.Group.create
    create.assert_called_once()
    comm_arg, cfg = create.call_args.args
    assert comm_arg is nccl_ep_fakes.torch_group_comm
    assert cfg.kwargs == {
        "algorithm": _Algorithm.LOW_LATENCY,
        "num_experts": _NUM_EXPERTS,
        "max_dispatch_tokens_per_rank": _MAX_TOKENS_PER_RANK,
        "max_recv_tokens_per_rank": mapping.moe_ep_size * _MAX_TOKENS_PER_RANK,
        "max_token_bytes": _HIDDEN_SIZE * 2,
    }
    nccl_ep_fakes.register.assert_called_once()
    assert nccl_ep_fakes.register.call_args.args[0]._group is nccl_ep_fakes.ep_group


@pytest.mark.parametrize("ep_world_rank", [0, 1])
def test_mpi_orchestrator_splits_session_comm_by_pipeline_stage(
    nccl_ep_fakes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, ep_world_rank: int
) -> None:
    monkeypatch.setenv("TLLM_DISABLE_MPI", "0")
    nccl_ep_fakes.sub_comm.Get_rank.return_value = ep_world_rank
    mapping = _make_mapping()
    mapping.moe_ep_rank = ep_world_rank

    ctx = _create_context(mapping)

    nccl_ep_fakes.mpi_comm.return_value.Split.assert_called_once_with(
        mapping.pp_rank, mapping.moe_ep_rank
    )
    nccl_ep_fakes.fake.interop_torch.get_nccl_comm_from_group.assert_not_called()
    if ep_world_rank == 0:
        nccl_ep_fakes.fake.core.get_unique_id.assert_called_once_with()
        nccl_ep_fakes.sub_comm.bcast.assert_called_once_with(_UNIQUE_ID, root=0)
    else:
        nccl_ep_fakes.fake.core.get_unique_id.assert_not_called()
        nccl_ep_fakes.sub_comm.bcast.assert_called_once_with(None, root=0)
    nccl_ep_fakes.fake.core.Communicator.init.assert_called_once_with(
        nranks=_EP_SIZE, rank=ep_world_rank, unique_id=_UNIQUE_ID
    )
    assert ctx._ep_mpi_comm is nccl_ep_fakes.sub_comm
    assert ctx.comm is nccl_ep_fakes.mpi_comm_result
    assert nccl_ep_fakes.fake.ep.Group.create.call_args.args[0] is nccl_ep_fakes.mpi_comm_result


# pytest adds each id as a keyword, and tests/unittest/conftest.py skips any item with a
# "ray" keyword unless --run-ray is given, so no id here may be exactly "ray".
@pytest.mark.parametrize("use_ray", [True, False], ids=["ray_orchestrator", "mpi_orchestrator"])
def test_destroy_releases_group_then_comm_and_frees_mpi_only_when_split(
    nccl_ep_fakes: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, use_ray: bool
) -> None:
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1" if use_ray else "0")
    ctx = _create_context(_make_mapping())
    comm = ctx.comm
    proxy = nccl_ep_fakes.register.call_args.args[0]
    order = Mock(name="teardown_order")
    order.attach_mock(nccl_ep_fakes.unregister, "unregister")
    order.attach_mock(nccl_ep_fakes.ep_group.destroy, "group_destroy")
    order.attach_mock(comm.finalize, "comm_finalize")
    order.attach_mock(comm.destroy, "comm_destroy")
    order.attach_mock(nccl_ep_fakes.sub_comm.Free, "mpi_free")

    ctx.destroy()

    # The timeout proxy must be unregistered before the group it points at is destroyed.
    expected = [
        call.unregister(proxy),
        call.group_destroy(),
        call.comm_finalize(),
        call.comm_destroy(),
    ]
    if not use_ray:
        expected.append(call.mpi_free())
    assert order.mock_calls == expected
    assert ctx.ep_group is None
    assert ctx.comm is None
    assert ctx._ep_mpi_comm is None
    assert ctx.output_tokens_buf is None
