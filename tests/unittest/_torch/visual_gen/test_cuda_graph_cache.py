# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound request-layout graph retention while preserving replay and live outputs."""

from unittest.mock import MagicMock, Mock

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
def test_capture_limit_does_not_count_tensor_keyword_with_callback_name(monkeypatch):
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    runner.register_extra_key_fn("layout", lambda *, layout, tag=None: tag, max_graphs=1)
    captures = []

    def capture(key, fn, args, kwargs):
        captures.append(key)
        runner.graphs[key] = MagicMock()
        runner.graph_outputs[key] = fn(*args, **kwargs)

    monkeypatch.setattr(runner, "capture", capture)
    monkeypatch.setattr(runner, "replay", lambda key, args, kwargs: runner.graph_outputs[key])
    wrapped = runner.wrap(lambda *, layout, tag=None: layout + (tag or 0))
    for size in (1, 2):
        wrapped(layout=torch.zeros(size))
    assert len(captures) == 2
    wrapped(layout=torch.zeros(1), tag=3)
    assert len(captures) == 3
    torch.testing.assert_close(wrapped(layout=torch.zeros(1), tag=4), torch.full((1,), 4.0))
    assert len(captures) == 3
    wrapped(layout=torch.zeros(3))
    assert len(captures) == 4


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.random.fork_rng()
def test_qwen_cfg_replay_and_eager_fallback_match_eager():
    """L2/T1: Qwen's CFG latent matches eager with one replayed and one uncached layout."""
    from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
    from tensorrt_llm._torch.visual_gen.models.qwen_image.request_batch import QwenImageBatchLayout
    from tensorrt_llm._torch.visual_gen.models.qwen_image.transformer_qwen_image import (
        QwenImageTransformer2DModel,
    )

    torch.manual_seed(0)
    model = (
        QwenImageTransformer2DModel(
            model_config=DiffusionModelConfig(
                attention={"backend": "VANILLA"}, skip_create_weights_in_init=False
            ),
            patch_size=1,
            in_channels=16,
            out_channels=16,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=16,
            axes_dims_rope=(4, 6, 6),
        )
        .to_inference_dtype()
        .cuda()
        .eval()
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.1)
    inputs = dict(
        hidden_states=torch.randn(2, 4, 16, device="cuda", dtype=torch.bfloat16),
        encoder_hidden_states=torch.randn(2, 3, 16, device="cuda", dtype=torch.bfloat16),
        timestep=torch.full((2,), 0.5, device="cuda", dtype=torch.bfloat16),
        img_shapes=[[(1, 2, 2)]],
    )
    conditional_layout = QwenImageBatchLayout((1, 1), (2, 3), (True, True))
    negative_layout = QwenImageBatchLayout((1, 1), (3, 2), (True, True))
    runner = CUDAGraphRunner(CUDAGraphRunnerConfig(use_cuda_graph=True))
    model.register_cuda_graph_extra_key_fns(runner)
    runner.register_extra_key_fn(
        "request_batch", lambda **kwargs: kwargs.get("request_batch"), max_graphs=1
    )
    calls = Mock(wraps=model.forward)
    wrapped = runner.wrap(calls)
    try:
        with torch.inference_mode():
            expected_cond = model(**inputs, request_batch=conditional_layout)[0]
            expected_neg = model(**inputs, request_batch=negative_layout)[0]
            expected = expected_neg + 3.5 * (expected_cond - expected_neg)
            wrapped(**inputs, request_batch=conditional_layout)
            calls.reset_mock()
            conditional = wrapped(**inputs, request_batch=conditional_layout)[0]
            calls.assert_not_called()
            negative = wrapped(**inputs, request_batch=negative_layout)[0]
            calls.assert_called_once()
            assert len(runner.graphs) == 1
            actual = negative + 3.5 * (conditional - negative)
        reference = expected.float().flatten()
        delta = (actual.float() - expected.float()).flatten()
        reference_norm = reference.norm().item()
        assert reference_norm > 0
        relative_l2 = delta.norm().item() / reference_norm
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), reference, dim=0
        ).item()
        max_abs = delta.abs().max().item()
        metrics = f"Qwen CFG BF16 {tuple(actual.shape)}: relative_l2={relative_l2}, cosine={cosine}, max_abs={max_abs}"
        print(metrics)
        assert relative_l2 <= 1e-2 and cosine >= 0.9999, metrics
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3, msg=metrics)
    finally:
        torch.cuda.synchronize()
        runner.clear()
