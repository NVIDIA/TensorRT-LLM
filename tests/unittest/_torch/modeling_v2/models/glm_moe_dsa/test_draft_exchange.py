# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of collective failure handling during draft workspace setup."""

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.spec import (
    _draft_exchange as exchange,
)

pytestmark = pytest.mark.cpu_only


class _Comm:
    def __init__(self, successes: int) -> None:
        self.successes = successes
        self.votes: list[int] = []

    def py2f(self) -> int:
        return 42

    def allreduce(self, value: int) -> int:
        self.votes.append(value)
        return self.successes

    def Get_size(self) -> int:
        return 2


class _Buffer:
    def get_uc_buffer(
        self, rank: int, shape: tuple[int, ...], dtype: torch.dtype, offset: int
    ) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype)


@pytest.fixture
def cpu_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SimpleNamespace, list[weakref.ReferenceType]]:
    handles: list[weakref.ReferenceType] = []

    def allocate(*args: object, **kwargs: object) -> _Buffer:
        handle = _Buffer()
        handles.append(weakref.ref(handle))
        return handle

    def cpu_tensor(data: list[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype)

    cpu_torch = SimpleNamespace(
        distributed=torch.distributed,
        int32=torch.int32,
        uint32=torch.uint32,
        device=torch.device,
        tensor=cpu_tensor,
        inference_mode=torch.inference_mode,
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(exchange, "torch", cpu_torch)
    monkeypatch.setattr(exchange, "McastGPUBuffer", allocate)
    return cpu_torch, handles


def _fail_at(
    monkeypatch: pytest.MonkeyPatch,
    cpu_torch: SimpleNamespace,
    stage: str,
    error_type: type[BaseException],
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error_type("injected setup failure")

    if stage == "allocate":
        monkeypatch.setattr(exchange, "McastGPUBuffer", fail)
    elif stage == "view":
        monkeypatch.setattr(_Buffer, "get_uc_buffer", fail)
    elif stage == "flags":
        monkeypatch.setattr(cpu_torch, "tensor", fail)
    else:
        monkeypatch.setattr(cpu_torch.cuda, "synchronize", fail)


@pytest.mark.parametrize("error_type", [MemoryError, OSError, TypeError])
@pytest.mark.parametrize("stage", ["allocate", "view", "flags", "synchronize"])
def test_setup_failure_reaches_collective(
    monkeypatch: pytest.MonkeyPatch, cpu_setup, stage: str, error_type: type[Exception]
) -> None:
    cpu_torch, handles = cpu_setup
    _fail_at(monkeypatch, cpu_torch, stage, error_type)
    comm = _Comm(successes=1)

    with pytest.raises(RuntimeError, match="failed on at least one rank") as caught:
        exchange.DraftExchangeWorkspace(16, 2, 0, 0, False, comm)

    assert comm.votes == [0]
    assert isinstance(caught.value.__cause__, error_type)
    assert str(caught.value.__cause__) == "injected setup failure"
    del caught
    gc.collect()
    assert all(handle() is None for handle in handles)


def test_peer_failure_rejects_successful_local_allocation(cpu_setup) -> None:
    _, handles = cpu_setup
    comm = _Comm(successes=1)

    with pytest.raises(RuntimeError, match="failed on at least one rank") as caught:
        exchange.DraftExchangeWorkspace(16, 2, 1, 0, False, comm)

    assert comm.votes == [1]
    assert caught.value.__cause__ is None
    del caught
    gc.collect()
    assert len(handles) == 1 and handles[0]() is None


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_exception_propagates(
    monkeypatch: pytest.MonkeyPatch, cpu_setup, error_type: type[BaseException]
) -> None:
    cpu_torch, handles = cpu_setup
    _fail_at(monkeypatch, cpu_torch, "flags", error_type)
    comm = _Comm(successes=1)

    with pytest.raises(error_type, match="injected setup failure") as caught:
        exchange.DraftExchangeWorkspace(16, 2, 0, 0, False, comm)

    assert comm.votes == []
    assert caught.value.__cause__ is None
    del caught
    gc.collect()
    assert len(handles) == 1 and handles[0]() is None


def test_success_retains_workspace_until_release(cpu_setup) -> None:
    _, handles = cpu_setup
    comm = _Comm(successes=2)
    workspace = exchange.DraftExchangeWorkspace(16, 2, 0, 0, False, comm)

    assert comm.votes == [1]
    assert workspace._comm is comm
    assert workspace._handle is handles[0]()
    assert workspace.buffer.tolist() == [-1] * 12
    assert workspace.flags.tolist() == [0, 2, 16, 0, 0, 0, 0, 0, 0]
    del workspace
    gc.collect()
    assert handles[0]() is None
