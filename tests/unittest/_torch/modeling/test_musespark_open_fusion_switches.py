# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When the model builds the fused finalize + all-reduce workspace, and how it degrades.

Host-side only: the workspace constructor and the device queries are replaced,
so these pin the selection without a GPU or IPC.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models import modeling_musespark_open as modeling
from tensorrt_llm._torch.models import musespark_open_finalize_ar as finalize_ar

pytestmark = pytest.mark.cpu_only

MAPPING = object()


@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []
    monkeypatch.setattr(modeling.logger, "warning", seen.append)
    return seen


@pytest.mark.parametrize("error", [ValueError("tp mismatch"), RuntimeError("peer not accessible")])
def test_a_failed_probe_falls_back_with_a_warning(monkeypatch, warnings_seen, error):
    """A group whose GPUs cannot map each other's memory must not fail model construction."""

    def raise_probe_failure(mapping):
        assert mapping is MAPPING
        raise error

    monkeypatch.setattr(finalize_ar, "HAS_CUTEDSL", True)
    monkeypatch.setattr(finalize_ar, "FinalizeIpcWorkspace", raise_probe_failure)
    assert modeling._build_finalize_allreduce_workspace(MAPPING) is None
    assert len(warnings_seen) == 1
    assert str(error) in warnings_seen[0]


def test_the_workspace_is_built_when_the_probe_succeeds(monkeypatch, warnings_seen):
    workspace = object()
    monkeypatch.setattr(finalize_ar, "HAS_CUTEDSL", True)
    monkeypatch.setattr(finalize_ar, "FinalizeIpcWorkspace", lambda mapping: workspace)
    assert modeling._build_finalize_allreduce_workspace(MAPPING) is workspace
    assert warnings_seen == []


def test_without_cute_dsl_nothing_is_probed(monkeypatch, warnings_seen):
    def must_not_construct(mapping):
        raise AssertionError("FinalizeIpcWorkspace must not be constructed without CuTe DSL")

    monkeypatch.setattr(finalize_ar, "HAS_CUTEDSL", False)
    monkeypatch.setattr(finalize_ar, "FinalizeIpcWorkspace", must_not_construct)
    assert modeling._build_finalize_allreduce_workspace(MAPPING) is None
    assert warnings_seen == []


def _mapping(**changes):
    values = dict(
        tp_size=4, gpus_per_node=4, tp_group=[0, 1, 2, 3], moe_ep_size=4, enable_attention_dp=False
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _layers(expert_hidden_size=4096, top_k=8, comm=None):
    moe = SimpleNamespace(
        expert_hidden_size=expert_hidden_size, top_k=top_k, experts=SimpleNamespace(comm=comm)
    )
    return [SimpleNamespace(feed_forward=moe) for _ in range(3)]


@pytest.fixture
def device(monkeypatch):
    """A GB300 unless a test says otherwise."""
    state = SimpleNamespace(available=True, capability=(10, 3))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: state.available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: state.capability)
    return state


BF16 = SimpleNamespace(torch_dtype=torch.bfloat16)


def test_bf16_tp4_ep4_on_one_gb300_node_is_supported(device):
    assert modeling._finalize_allreduce_supported(_mapping(), BF16, _layers())


@pytest.mark.parametrize(
    "mapping",
    [
        _mapping(tp_size=8, tp_group=list(range(8)), moe_ep_size=8),
        _mapping(tp_size=2, tp_group=[0, 1], moe_ep_size=2),
        # Four ranks over two nodes.
        _mapping(tp_group=[0, 1, 4, 5]),
        _mapping(moe_ep_size=1),
        _mapping(enable_attention_dp=True),
    ],
    ids=["tp8", "tp2", "two_nodes", "moe_tp", "attention_dp"],
)
def test_other_parallel_layouts_are_not(device, mapping):
    assert not modeling._finalize_allreduce_supported(mapping, BF16, _layers())


@pytest.mark.parametrize(
    "layers",
    [_layers(expert_hidden_size=2048), _layers(top_k=4), _layers(comm=object())],
    ids=["expert_width", "top_k", "moe_comm"],
)
def test_other_expert_shapes_and_communication_are_not(device, layers):
    assert not modeling._finalize_allreduce_supported(_mapping(), BF16, layers)


def test_other_dtypes_and_devices_are_not(device):
    fp16 = SimpleNamespace(torch_dtype=torch.float16)
    assert not modeling._finalize_allreduce_supported(_mapping(), fp16, _layers())
    device.capability = (10, 0)
    assert not modeling._finalize_allreduce_supported(_mapping(), BF16, _layers())
    device.available = False
    assert not modeling._finalize_allreduce_supported(_mapping(), BF16, _layers())
