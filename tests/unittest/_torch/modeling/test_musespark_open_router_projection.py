# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-batch FP32 router projection, including mutable CUDA-graph inputs."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenMoE


def _module(device, hidden=8192, experts=256):
    module = MuseSparkOpenMoE.__new__(MuseSparkOpenMoE)
    torch.nn.Module.__init__(module)
    module.router = torch.nn.Linear(hidden, experts, bias=False, device=device, dtype=torch.float32)
    module.metap_multiplier = 0.375
    return module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 16])
@torch.no_grad()
def test_router_matches_fp64_and_replays_changed_inputs(tokens):
    torch.manual_seed(1263 + tokens)
    module = _module("cuda")
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    module.router_logits(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.router_logits(x)
    for factor in (1.0, -0.5, 2.0):
        x.normal_().mul_(factor)
        # Verify graph reads live weights too, rather than a stale cast/cache.
        module.router.weight.mul_(-0.75)
        graph.replay()
        expected = (
            x.double() @ module.router.weight.double().t() * module.metap_multiplier
        ).float()
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)
        assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [17, 32, 48, 80, 128])
@torch.no_grad()
def test_tensorcore_router_partial_tiles_and_graph_replay(tokens):
    torch.manual_seed(1327 + tokens)
    module = _module("cuda")
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    module.router_logits(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.router_logits(x)
    for factor in (1.0, -0.5, 2.0):
        x.normal_().mul_(factor)
        module.router.weight.mul_(-0.75)
        graph.replay()
        expected = (
            x.double() @ module.router.weight.double().t() * module.metap_multiplier
        ).float()
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)


@torch.no_grad()
def test_cpu_fallback_calls_linear():
    module = _module("cpu", hidden=16, experts=8)
    x = torch.randn(2, 16, dtype=torch.bfloat16)
    calls = []
    hook = module.router.register_forward_hook(lambda *_: calls.append(True))
    actual = module.router_logits(x)
    hook.remove()
    assert calls == [True]
    expected = torch.nn.functional.linear(x.float(), module.router.weight) * module.metap_multiplier
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [2, 8, 9, 16, 128, 129])
@torch.no_grad()
def test_router_automatic_small_batch_dispatch(tokens):
    module = _module("cuda")
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    calls = []
    hook = module.router.register_forward_hook(lambda *_: calls.append(True))
    output = module.router_logits(x)
    hook.remove()
    assert len(calls) == int(tokens > 128)
    if tokens > 128:
        expected = (
            torch.nn.functional.linear(x.float(), module.router.weight) * module.metap_multiplier
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    else:
        expected = (
            x.double() @ module.router.weight.double().t() * module.metap_multiplier
        ).float()
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens,noncontiguous", [(0, False), (129, False), (16, True)])
@torch.no_grad()
def test_unsupported_gpu_inputs_use_linear(tokens, noncontiguous):
    module = _module("cuda")
    width = 16384 if noncontiguous else 8192
    x = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
    if noncontiguous:
        x = x[:, ::2]
    calls = []
    hook = module.router.register_forward_hook(lambda *_: calls.append(True))
    actual = module.router_logits(x)
    hook.remove()
    assert calls == [True]
    expected = torch.nn.functional.linear(x.float(), module.router.weight) * module.metap_multiplier
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", range(1, 17))
@torch.no_grad()
def test_overlap_tuned_router_reads_mutable_graph_inputs(tokens: int) -> None:
    torch.manual_seed(1291 + tokens)
    module = _module("cuda")
    module._prepare_stream = torch.cuda.Stream()
    module.experts = SimpleNamespace(comm=None)
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    module.router_logits(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.router_logits(x)
    for factor in (1.0, -0.5, 2.0):
        x.normal_().mul_(factor)
        module.router.weight.mul_(-0.75)
        graph.replay()
        expected = (
            x.double() @ module.router.weight.double().t() * module.metap_multiplier
        ).float()
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [256, 512, 4096, 8193, 32768])
@torch.no_grad()
def test_bf16x3_router_large_batch_mutable_graph(tokens: int) -> None:
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Validated on GB300")
    torch.manual_seed(1927)
    module = _module("cuda")
    module._bf16x3_router_enabled = True
    inputs = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    module.router_logits(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.router_logits(inputs)
    rows = torch.linspace(0, tokens - 1, min(tokens, 512), device="cuda").long()
    for factor in (1.0, -0.5, 2.0):
        inputs.normal_().mul_(factor)
        module.router.weight.mul_(-0.75)
        graph.replay()
        expected = (
            inputs[rows].double() @ module.router.weight.double().T * module.metap_multiplier
        ).float()
        torch.testing.assert_close(output[rows], expected, rtol=2e-5, atol=2e-6)
