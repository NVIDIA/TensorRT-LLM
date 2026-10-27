# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The X1 router payload reaches the real expert call without changing fallback routing."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    modeling as m,
)


@pytest.mark.parametrize("rows", [9, 16, 24, 32, 48])
@pytest.mark.parametrize("scale_value", [0.125, 1.0, 336.0, 4096.0])
@pytest.mark.parametrize("kind", ["random", "zero", "tiny", "large", "midpoints"])
def test_native_payload_exact_eager_and_graph(rows, scale_value, kind):
    """Donor payload contract, without a rounding-tie tolerance, against today's quantizer."""
    assert torch.cuda.get_device_capability() == (10, 7)
    gen = torch.Generator(device="cuda").manual_seed(20623 + rows)
    x = torch.randn(rows, 6144, device="cuda", generator=gen).to(torch.bfloat16)
    w = torch.randn(128, 6144, device="cuda", generator=gen) / 6144**0.5
    scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")
    if kind == "zero":
        x.zero_()
        x[:, 1::2] = -0.0
    elif kind == "tiny":
        x.mul_(2**-12)
    elif kind == "large":
        x.mul_(2**12)
    elif kind == "midpoints":
        values = torch.tensor(
            [0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5],
            device="cuda",
            dtype=torch.bfloat16,
        )
        x.copy_(values.repeat(rows, 384))
    native = torch.ops.trtllm.minimax_m3_router_tc
    ref = [native(x, w, None, True)[0], *m.fp4_quantize(x, scale, 16, False, False)]
    actual = native(x, w, scale, True)
    assert len(actual) == len(ref) == 3
    for a, b in zip(actual, ref):
        assert torch.equal(a.view(torch.uint8).flatten(), b.view(torch.uint8).flatten())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = native(x, w, scale, True)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(captured, ref):
            assert torch.equal(a.view(torch.uint8).flatten(), b.view(torch.uint8).flatten())


@pytest.mark.parametrize("rows", [8, 9, 16, 24, 32, 48, 49, 176])
@pytest.mark.parametrize("cluster", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_payload_and_fallback_consumer(monkeypatch, rows, cluster, enabled, finalize):
    h = torch.zeros(rows, 64, dtype=torch.bfloat16)
    scale = torch.ones(1)
    logits = torch.zeros(rows, 128)
    q = torch.zeros(rows, 32, dtype=torch.uint8)
    sf = torch.zeros(rows * 4, dtype=torch.uint8)
    core = SimpleNamespace(
        s1b_pins={},
        router_tc_cluster16=cluster,
        w={"l3_router": torch.zeros(128, 64)},
        moe={3: SimpleNamespace(fc31_input_scale=scale)},
        moe_max_num_tokens=16384,
    )
    calls = []

    def router(x, weight, input_scale, pdl):
        assert x is h and weight is core.w["l3_router"]
        calls.append(("router", input_scale))
        return [logits] if input_scale is None else [logits, q, sf]

    def experts(actual_core, layer, x, x_sf, actual_logits, do_finalize, topk=None):
        assert actual_core is core and layer == 3 and actual_logits is logits
        calls.append(("experts", x, x_sf, do_finalize))
        return "experts"

    target = SimpleNamespace(
        core=core,
        _routed_chunks=lambda layer, x, router_logits, topk=None: calls.append(("chunks", x))
        or "chunks",
    )
    monkeypatch.setattr(m, "X1", enabled)
    monkeypatch.setattr(m, "m3_router_tc", router)
    monkeypatch.setattr(m, "m3_fp32_linear", lambda *args: logits)
    monkeypatch.setattr(m, "_routed_experts", experts)
    monkeypatch.setattr(m, "taken", lambda *args: None)
    # The parked decode-router-F32 slot is absent from the frozen vehicle.
    if hasattr(m, "_DECODE_ROUTER_F32"):
        m._DECODE_ROUTER_F32.clear()
    out = m.DecodeTarget._routed(target, 3, h, finalize)
    fused = enabled and cluster and 9 <= rows <= m.ROUTER_TC_GEN_MAX_ROWS
    if fused:
        assert calls[0][1] is scale
        assert calls[-1][1] is q and calls[-1][2] is sf
        assert calls[-1][3] is finalize and out == "experts"
    elif finalize:
        assert calls[-1][0] == "chunks" and calls[-1][1] is h
        assert out == "chunks"
    else:
        assert calls[-1][1] is h and calls[-1][2] is None
        assert calls[-1][3] is False and out == "experts"


@pytest.mark.parametrize("rows", [9, 16, 24, 32, 48])
@pytest.mark.parametrize("finalize", [False, True])
def test_real_experts_match_current_path_eager_and_graph(monkeypatch, rows, finalize):
    """Donor 780d57e886 payload/expert contract through M31's actual routed call."""
    from _m3_moe_operands import experts, hidden_states

    from tensorrt_llm._torch.autotuner import AutoTuner
    from tensorrt_llm.mapping import Mapping

    assert torch.cuda.get_device_capability() == (10, 7)
    assert torch.ops.trtllm.minimax_m3_router_tc_cluster16(6144)
    AutoTuner.get().setup_distributed_state(Mapping())
    ops = experts()
    w31, s31, w2, s2, a31, scale_c, act_a, act_b, act_c, a2 = ops.tensors
    core = SimpleNamespace(
        s1b_pins={},
        router_tc_cluster16=True,
        w={
            "l3_router": ops.router_weight,
            "l3_router_bias": ops.bias,
            "l3_e_w31": w31,
            "l3_e_s31": s31,
            "l3_e_w2": w2,
            "l3_e_s2": s2,
        },
        moe={
            3: SimpleNamespace(
                fc31_input_scale=ops.fc31_input_scale,
                act_alpha=act_a,
                act_beta=act_b,
                act_clamp=act_c,
                fc31_scale_c=scale_c,
                fc31_alpha=a31,
                fc2_alpha=a2,
            )
        },
        mxfp8=SimpleNamespace(
            quant_method=SimpleNamespace(
                tune_decode_graph_backends=False,
                backend="auto",
                _flashinfer_autotuned=False,
                needs_native_autotune=False,
            )
        ),
        moe_max_num_tokens=16384,
        max_num_tokens=16384,
        num_experts=128,
        top_k=4,
        moe_inter=1536,
        routed_scaling_factor=2.0,
    )
    target = object.__new__(m.DecodeTarget)
    target.core = core
    h = hidden_states(rows, 20623 + rows)
    # The parked decode-router-F32 slot is absent from the frozen vehicle.
    if hasattr(m, "_DECODE_ROUTER_F32"):
        m._DECODE_ROUTER_F32.clear()

    def call():
        output = target._routed(3, h, finalize)
        return output if finalize else m._triton_finalize(rows, output, None)

    monkeypatch.setattr(m, "X1", False)
    reference = call()
    monkeypatch.setattr(m, "X1", True)
    actual = call()
    assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.uint8), reference.view(torch.uint8))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_x1_preserves_wide_s1b_topk(monkeypatch, enabled, finalize):
    """A wide S1b route must retain its top-k payload with either X1 policy."""
    h = torch.zeros(176, 64, dtype=torch.bfloat16)
    topk = (torch.zeros(176, 4, dtype=torch.int32), torch.ones(176, 4))
    core = SimpleNamespace(moe_max_num_tokens=448)

    def route(actual_core, layer, x, fp4=None):
        assert actual_core is core and layer == 3 and x is h and fp4 is None
        return "topk", topk

    def experts(actual_core, layer, x, sf, logits, do_finalize, actual_topk=None):
        assert actual_core is core and x is h and sf is None and logits is None
        assert not do_finalize and actual_topk is topk
        return topk

    def chunks(layer, x, logits, actual_topk=None):
        assert layer == 3 and x is h and logits is None and actual_topk is topk
        return topk

    target = SimpleNamespace(core=core, _routed_chunks=chunks)
    monkeypatch.setattr(m, "X1", enabled)
    monkeypatch.setattr(m, "S1B_FOLD_CHECK", False)
    monkeypatch.setattr(m, "_decode_route", route)
    monkeypatch.setattr(m, "_routed_experts", experts)
    assert m.DecodeTarget._routed(target, 3, h, finalize) is topk
