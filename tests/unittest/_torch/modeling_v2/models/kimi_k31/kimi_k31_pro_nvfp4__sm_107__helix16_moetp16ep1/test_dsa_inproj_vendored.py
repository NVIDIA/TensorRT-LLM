# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the 17–24-row packed DSA in-projection in eager mode and CUDA graphs."""

import importlib

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"


def _modeling():
    return importlib.import_module(f"{_PKG}.modeling")


def test_the_table() -> None:
    modeling = _modeling()
    assert modeling.DSA_INPROJ_ROWS == range(17, 25)
    assert modeling.DSA_INPROJ_TACTIC == (64, 24, 2, 10)
    # The plan these rows replace (same split, so the same FP32 summation order).
    assert dict(modeling.ULL_DECODE_PLAN[(3040, 7168)])[32] == ("splitk", 64, 32, 2, 9)
    # The 3-16 rows stay on the warp split-K (and 1-2 on the plan's direct kernel).
    assert max(modeling.ROWS) == 16


def test_cpu_prepares_nothing() -> None:
    w = torch.zeros(3040, 7168, dtype=torch.bfloat16)
    engine = _modeling().DsaInProjVendored(w)
    engine.prepare()
    assert engine.projection is None
    assert not engine.supports(torch.zeros(24, 7168, dtype=torch.bfloat16))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="compiles the vendored and FlashInfer split-K for an SM10x GPU",
)
def test_bitwise_the_plan_eager_and_graph() -> None:
    modeling = _modeling()
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=dev).manual_seed(1117)
    w = (torch.randn(3040, 7168, device=dev, generator=gen) * 0.02).bfloat16()
    plan = modeling.Bf16Weight(w)
    engine = modeling.DsaInProjVendored(w)
    engine.prepare()
    assert sorted(engine.projection.kernels) == list(range(17, 25))
    for rows in (8, 16, 25, 32, 33):
        assert not engine.supports(torch.zeros(rows, 7168, dtype=torch.bfloat16, device=dev))
    for rows in (17, 20, 24):
        x = torch.randn(rows, 7168, device=dev, generator=gen).bfloat16()
        assert engine.supports(x)
        want = modeling.bf16_projection(x, plan)
        got = engine(x)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.uint16), want.view(torch.uint16)), rows
    # A captured graph replays the same launch on new inputs.
    x = torch.randn(24, 7168, device=dev, generator=gen).bfloat16()
    engine(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = engine(x)
    x.copy_(torch.randn(24, 7168, device=dev, generator=gen).bfloat16())
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.uint16), modeling.bf16_projection(x, plan).view(torch.uint16))
