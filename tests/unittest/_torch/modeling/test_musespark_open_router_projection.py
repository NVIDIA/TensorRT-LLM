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
    module._prepare_stream = None
    module.experts = SimpleNamespace(comm=None)
    return module


def _fp64_reference(module, x):
    return (x.double() @ module.router.weight.double().t() * module.metap_multiplier).float()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", [1, 2, 3, 8, 15, 16, 17, 32, 48, 80, 128])
@torch.no_grad()
def test_router_matches_fp64_and_replays_changed_inputs(tokens):
    """Both the CUDA-core (under 16 tokens) and the tensor-core kernel."""
    torch.manual_seed(1263 + tokens)
    module = _module("cuda")
    x = torch.randn(tokens, 8192, device="cuda", dtype=torch.bfloat16)
    calls = []
    hook = module.router.register_forward_hook(lambda *_: calls.append(True))
    module.router_logits(x)
    hook.remove()
    assert calls == [], "a supported batch fell back to the Linear"
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module.router_logits(x)
    for factor in (1.0, -0.5, 2.0):
        x.normal_().mul_(factor)
        # The replay must read the live weights too, not a cast or a cache.
        module.router.weight.mul_(-0.75)
        graph.replay()
        torch.testing.assert_close(output, _fp64_reference(module, x), rtol=2e-5, atol=2e-6)
        assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tokens", range(1, 17))
@torch.no_grad()
def test_overlap_tuned_router_reads_mutable_graph_inputs(tokens: int) -> None:
    """The launch shapes used while the pooler runs on the auxiliary stream."""
    torch.manual_seed(1291 + tokens)
    module = _module("cuda")
    module._prepare_stream = torch.cuda.Stream()
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
        torch.testing.assert_close(output, _fp64_reference(module, x), rtol=2e-5, atol=2e-6)


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
