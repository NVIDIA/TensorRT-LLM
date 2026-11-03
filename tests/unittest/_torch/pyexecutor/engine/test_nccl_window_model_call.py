# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound eager scope work by activation arguments, preserving enclosing ownership."""

from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses import FakeTensorMode

from tensorrt_llm._torch.nccl_window_tensor_scope import nccl_window_tensor_scoped_module
from tensorrt_llm._torch.pyexecutor.model_engine import ModelEngine
from tensorrt_llm._torch.utils import model_extra_attrs, torch_compiling
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class _RequestIds(list[int]):
    visits = 0

    def __iter__(self) -> Iterator[int]:
        self.visits += 1
        return super().__iter__()


@dataclass
class _Metadata:
    seq_lens: torch.Tensor
    request_ids: _RequestIds


@nccl_window_tensor_scoped_module
class _Layer(torch.nn.Module):
    def forward(
        self, hidden: torch.Tensor, values: torch.Tensor, *, attn_metadata: _Metadata, fail: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fail:
            raise RuntimeError("layer failed")
        return hidden + 1, values + 1


@nccl_window_tensor_scoped_module
class _Stack(torch.nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(num_layers))

    def forward(
        self, hidden: torch.Tensor, values: torch.Tensor, *, attn_metadata: _Metadata, fail: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            hidden, values = layer(hidden, values, attn_metadata=attn_metadata, fail=fail)
        return hidden, values


class _Model(torch.nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.model_config = SimpleNamespace(extra_attrs={})
        self.stack = _Stack(num_layers)

    def forward(self, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return self.stack(**kwargs)


@pytest.fixture
def scope_calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple]]:
    """Observe the native boundary while retaining the real discovery and hook code."""
    calls = []
    active = []

    def begin(inputs: list[torch.Tensor]) -> int:
        depth = len(active)
        identities = tuple(id(tensor) for tensor in inputs)
        active.append(identities)
        calls.append(("begin", identities, depth))
        return depth

    def end(
        inputs: list[torch.Tensor], outputs: list[torch.Tensor], failed: bool, depth: int
    ) -> None:
        assert len(active) == depth + 1
        identities = tuple(id(tensor) for tensor in inputs)
        assert active.pop() == identities
        calls.append(("end", identities, tuple(id(tensor) for tensor in outputs), failed, depth))

    monkeypatch.setattr(torch.ops.trtllm.begin_nccl_window_tensor_scope, "tracked", begin)
    monkeypatch.setattr(torch.ops.trtllm.end_nccl_window_tensor_scope, "tracked", end)
    yield calls
    assert not active


def _compile_engine(peer_mode: str, num_layers: int) -> ModelEngine:
    # Only compilation setup is under test; no checkpoint, cache allocation or GPU startup.
    engine = object.__new__(ModelEngine)
    engine._cleanup_done = True
    engine.model = _Model(num_layers)
    world_size = 2 if peer_mode in ("no-communicator", "peers") else 1
    engine.mapping = Mapping(
        world_size=world_size, tp_size=world_size, dwdp_size=2 if peer_mode == "dwdp" else 0
    )
    engine.dist = None if peer_mode == "no-communicator" else SimpleNamespace(world_size=world_size)
    engine.llm_args = SimpleNamespace(
        torch_compile_config=None,
        prefill_cuda_graph_backend=None,
        allreduce_strategy="AUTO",
        multimodal_config=None,
    )
    engine._torch_compile_backend = None
    with torch_compiling(False):
        engine._compile_model(None)
    return engine


@pytest.mark.parametrize("peer_mode", ["no-communicator", "single-rank", "peers", "dwdp"])
@pytest.mark.parametrize("num_layers,num_requests", [(1, 2), (3, 7)])
def test_engine_scopes_traverse_metadata_only_with_peers_at_outer_boundary(
    scope_calls: list[tuple], peer_mode: str, num_layers: int, num_requests: int
) -> None:
    engine = _compile_engine(peer_mode, num_layers)
    with FakeTensorMode():
        hidden = torch.empty(2, device="cuda")
        seq_lens = torch.empty(num_requests, device="cuda")
    metadata = _Metadata(seq_lens, _RequestIds(range(num_requests)))
    values = torch.arange(2)

    try:
        with model_extra_attrs({}):
            output, actual_values = engine._model_caller(
                hidden=hidden, values=values, attn_metadata=metadata, fail=False
            )
    finally:
        for handle in engine._nccl_window_tensor_scope_hooks:
            handle.remove()

    torch.testing.assert_close(actual_values, values + num_layers)
    if peer_mode in ("no-communicator", "single-rank"):
        assert scope_calls == [], "A forward without peers must not enter native lease scopes"
        assert metadata.request_ids.visits == 0
        assert engine._nccl_window_tensor_scope_hooks == []
        return

    assert metadata.request_ids.visits == 1, "Only the enclosing forward owns shared metadata"
    begins = [call for call in scope_calls if call[0] == "begin"]
    ends = [call for call in scope_calls if call[0] == "end"]
    assert len(begins) == num_layers + 2  # Caller, marked stack, and its layers.
    assert begins[0][1] == (id(hidden), id(seq_lens))
    assert all(id(seq_lens) not in call[1] for call in begins[1:])
    assert [call[2] for call in begins] == [0, 1] + [2] * num_layers
    assert [call[-1] for call in ends] == [2] * num_layers + [1, 0]
    assert all(call[3] is False for call in ends)
    # Each producer's escaped tensor is adopted by the following layer.
    assert begins[2][1] == (id(hidden),)
    assert [call[1] for call in begins[3:]] == [call[2] for call in ends[: num_layers - 1]]
    assert all(call[2] == (id(output),) for call in ends[-3:])


def test_engine_scopes_unwind_nested_failure_before_next_forward(scope_calls: list[tuple]) -> None:
    engine = _compile_engine("peers", 1)
    with FakeTensorMode():
        hidden = torch.empty(2, device="cuda")
        seq_lens = torch.empty(1, device="cuda")
    metadata = _Metadata(seq_lens, _RequestIds([1]))
    values = torch.tensor([2, 5])
    try:
        with model_extra_attrs({}):
            with pytest.raises(RuntimeError, match="layer failed"):
                engine._model_caller(
                    hidden=hidden, values=values, attn_metadata=metadata, fail=True
                )
            failed = list(scope_calls)
            scope_calls.clear()
            output, actual_values = engine._model_caller(
                hidden=hidden, values=values, attn_metadata=metadata, fail=False
            )
    finally:
        for handle in engine._nccl_window_tensor_scope_hooks:
            handle.remove()

    torch.testing.assert_close(actual_values, torch.tensor([3, 6]))
    failed_ends = [call for call in failed if call[0] == "end"]
    assert [(call[2], call[3], call[4]) for call in failed_ends] == [
        ((), True, 2),
        ((), True, 1),
        ((), True, 0),
    ]
    assert scope_calls[0][2] == 0
    assert scope_calls[-1][2:] == ((id(output),), False, 0)
    assert metadata.request_ids.visits == 2
