# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound request-layout graph retention while preserving replay and live outputs."""

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner, CUDAGraphRunnerConfig


@pytest.mark.cpu_only
def test_capture_limit_bounds_all_retained_entries(monkeypatch):
    """New layouts fall back eagerly without evicting graphs or growing their buffers."""
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    runner.register_extra_key_fn("layout", lambda *, layout=None: layout, max_graphs=2)
    captures, replays, eager_calls = [], [], []

    def forward(*, layout=None):
        eager_calls.append(layout)
        return object()

    def capture(key, fn, args, kwargs):
        captures.append(key)
        runner.graphs[key] = MagicMock()
        runner.static_inputs[key] = (args, kwargs.copy())
        runner._graph_output_refs[key] = forward(**kwargs)
        runner.graph_outputs[key] = runner._graph_output_refs[key]

    def replay(key, args, kwargs):
        replays.append(key)
        return runner.graph_outputs[key]

    monkeypatch.setattr(runner, "capture", capture)
    monkeypatch.setattr(runner, "replay", replay)
    wrapped = runner.wrap(forward)
    first = wrapped(layout=0)
    second = wrapped(layout=1)
    for layout in range(2, 20):
        assert wrapped(layout=layout) is not first
    assert wrapped(layout=0) is first
    assert wrapped(layout=1) is second
    assert len(captures) == 2
    assert eager_calls == list(range(20))
    assert len(replays) == 4
    assert all(graph.reset.call_count == 0 for graph in runner.graphs.values())
    retained = [
        runner.graphs,
        runner.static_inputs,
        runner.graph_outputs,
        runner._graph_output_refs,
    ]
    assert all(len(entries) == 2 for entries in retained)

    # Calls without a layout retain the existing unrestricted singleton behavior.
    wrapped()
    assert len(captures) == 3
    graphs = list(runner.graphs.values())
    runner.clear()
    assert all(not entries for entries in retained)
    assert all(graph.reset.call_count == 1 for graph in graphs)
    wrapped(layout=20)
    assert len(captures) == 4


@pytest.mark.cpu_only
def test_capture_limit_respects_disallow_capture(monkeypatch):
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    runner.register_extra_key_fn("layout", lambda **kwargs: kwargs["layout"], max_graphs=1)
    runner.allow_capture = False
    capture = MagicMock()
    monkeypatch.setattr(runner, "capture", capture)
    assert runner.wrap(lambda *, layout: layout)(layout=5) == 5
    capture.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("limit", [0, -1])
def test_capture_limit_rejects_nonpositive_capacity(limit):
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    with pytest.raises(ValueError, match="positive"):
        runner.register_extra_key_fn("layout", lambda: 1, max_graphs=limit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capture_limit_preserves_live_cuda_outputs():
    """A new CFG layout cannot invalidate a previously returned graph output."""
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    runner.register_extra_key_fn("layout", lambda value, *, layout: layout, max_graphs=2)
    wrapped = runner.wrap(lambda value, *, layout: value.sin() + layout)
    value = torch.arange(32, dtype=torch.float32, device="cuda")
    try:
        first = wrapped(value, layout=0)
        second = wrapped(value, layout=1)
        for layout in range(2, 8):
            result = wrapped(value, layout=layout)
            torch.testing.assert_close(result, value.sin() + layout)
            torch.testing.assert_close(first, value.sin())
            torch.testing.assert_close(second, value.sin() + 1)
        assert len(runner.graphs) == 2
        torch.testing.assert_close(wrapped(value + 1, layout=0), (value + 1).sin())
        torch.testing.assert_close(second, value.sin() + 1)
    finally:
        torch.cuda.synchronize()
        runner.clear()
