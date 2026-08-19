# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest

from tensorrt_llm._torch import nccl_window_graph

pytestmark = pytest.mark.cpu_only


def test_pool_owner_is_value_based():
    assert nccl_window_graph._shared_pool_owner(
        (12345, 67890)
    ) == nccl_window_graph._shared_pool_owner(tuple([12345, 67890]))


def test_pool_owner_rejects_invalid_handles():
    with pytest.raises(TypeError, match="CUDA graph pool handle"):
        nccl_window_graph._shared_pool_owner(None)
    with pytest.raises(TypeError, match="Invalid CUDA graph pool handle"):
        nccl_window_graph._shared_pool_owner((1, "not-an-int"))


def test_release_removes_owner_after_backend_release(monkeypatch):
    pool = (23456, 78901)
    owner = nccl_window_graph._shared_pool_owner(pool)
    released = []
    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "release_nccl_window_graph_owner",
        released.append,
    )

    nccl_window_graph.release_nccl_window_graph_owner(pool)

    replacement_owner = nccl_window_graph._shared_pool_owner(pool)
    assert released == [owner]
    assert replacement_owner > owner


def test_release_retains_owner_when_backend_release_fails(monkeypatch):
    pool = (34567, 89012)
    owner = nccl_window_graph._shared_pool_owner(pool)

    def fail_release(_owner):
        raise RuntimeError("backend release failed")

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "release_nccl_window_graph_owner",
        fail_release,
    )

    with pytest.raises(RuntimeError, match="backend release failed"):
        nccl_window_graph.release_nccl_window_graph_owner(pool)

    assert nccl_window_graph._shared_pool_owner(pool) == owner


def test_graph_capture_sets_and_restores_owner(monkeypatch):
    pool = (45678, 90123)
    graph = object()
    owner = nccl_window_graph._shared_pool_owner(pool)
    events = []

    def set_owner(value):
        events.append(("owner", value))

    @contextlib.contextmanager
    def capture(captured_graph, *, pool, **kwargs):
        events.append(("capture", captured_graph, pool, kwargs))
        try:
            yield
        finally:
            events.append(("capture_exit",))

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        set_owner,
    )
    monkeypatch.setattr(nccl_window_graph.torch.cuda, "graph", capture)

    with nccl_window_graph.nccl_window_graph_capture(
        graph, pool, capture_error_mode="thread_local"
    ):
        events.append(("body",))

    assert events == [
        ("owner", owner),
        ("capture", graph, pool, {"capture_error_mode": "thread_local"}),
        ("body",),
        ("capture_exit",),
        ("owner", nccl_window_graph._EAGER_OWNER),
    ]


def test_graph_capture_restores_owner_when_capture_fails(monkeypatch):
    pool = (56789, 1234)
    graph = object()
    owner = nccl_window_graph._shared_pool_owner(pool)
    events = []

    def set_owner(value):
        events.append(("owner", value))

    def capture(captured_graph, *, pool, **kwargs):
        events.append(("capture", captured_graph, pool, kwargs))
        raise RuntimeError("capture failed")

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        set_owner,
    )
    monkeypatch.setattr(nccl_window_graph.torch.cuda, "graph", capture)

    with pytest.raises(RuntimeError, match="capture failed"):
        with nccl_window_graph.nccl_window_graph_capture(graph, pool):
            pytest.fail("capture body should not run")

    assert events == [
        ("owner", owner),
        ("capture", graph, pool, {}),
        ("owner", nccl_window_graph._EAGER_OWNER),
    ]


def test_nested_graph_capture_restores_previous_owner(monkeypatch):
    outer_pool = (67890, 2345)
    inner_pool = (78901, 3456)
    outer_owner = nccl_window_graph._shared_pool_owner(outer_pool)
    inner_owner = nccl_window_graph._shared_pool_owner(inner_pool)
    owners = []

    monkeypatch.setattr(
        nccl_window_graph.torch.ops.trtllm,
        "set_nccl_window_graph_owner",
        owners.append,
    )
    monkeypatch.setattr(
        nccl_window_graph.torch.cuda,
        "graph",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )

    with nccl_window_graph.nccl_window_graph_capture(object(), outer_pool):
        with nccl_window_graph.nccl_window_graph_capture(object(), inner_pool):
            pass

    assert owners == [
        outer_owner,
        inner_owner,
        outer_owner,
        nccl_window_graph._EAGER_OWNER,
    ]
