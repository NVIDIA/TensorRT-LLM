# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The TP8 MoE engine's completion ring on SM107: an oracle and repeated-call / graph checks.

This component test imports the real TP8 engine. A zero-valued shared expert
allows an independent TRTLLM-Gen routed-output oracle while exercising shared
CTA synchronization. Nonzero shared weights additionally check eager/graph
repeatability; that check is not an independent shared-expert numeric oracle.
Run under a subprocess timeout: a liveness failure must fail the test.
"""

import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

import tensorrt_llm  # noqa: F401 -- register native ops before importing the engine

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp8"
PKG = f"tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.{TARGET}"
SM107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)


@pytest.fixture(scope="module")
def runtime():
    path = Path(__file__).parent / "moe/_m3_moe_operands.py"
    spec = importlib.util.spec_from_file_location("_tp8_ring_operands", path)
    operands = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operands)
    return (
        importlib.import_module(f"{PKG}.kernels.m3_moe.op"),
        importlib.import_module(f"{PKG}.ops.moe.m3_moe_engine"),
        operands,
    )


def _structured_inputs(operands, rows, layer):
    # At eight rows this is exactly 29 distinct experts / eight groups. At
    # 188 routed CTAs, the production phase plan has six idle FC1 groups on CTA2.
    groups = (
        (0, 1, 2, 3),
        (0, 4, 5, 6),
        (1, 7, 8, 9),
        (2, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
        (21, 22, 23, 24),
        (25, 26, 27, 28),
    )
    hidden = torch.zeros(rows, operands.H, dtype=torch.bfloat16, device="cuda")
    hidden[:, :rows].copy_(torch.eye(rows, device="cuda", dtype=torch.bfloat16))
    router = torch.zeros_like(layer.router_weight)
    router[:, :rows] = -4.0
    for row, experts in enumerate(groups[:rows]):
        router[list(experts), row] = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda")
    layer.router_weight = router
    layer.bias = torch.zeros_like(layer.bias)
    return hidden, groups[:rows]


def _assert_routed_oracle(state, provider, operands, hidden, layer, expected_ids=None):
    output = state.run(layer, hidden).clone()
    torch.cuda.synchronize()
    ids = state.route_dbg[:, :4].long().clone()
    weights = state.route_dbg[:, 4:].contiguous().view(torch.float32).clone()
    ref_ids, ref_weights = provider.moe_route(hidden.float(), layer.router_weight, layer.bias)
    assert torch.equal(ids, ref_ids), "routing ids differ from the independent route reference"
    assert torch.equal(weights, ref_weights), "routing weights differ from the route reference"
    if expected_ids is not None:
        expected = torch.tensor(expected_ids, device="cuda", dtype=ids.dtype)
        assert torch.equal(ids, expected), "structured input did not exercise its promised routing"
        if hidden.shape[0] == 8:
            assert ids.unique().numel() == 29, "missing 29-expert idle-group witness"
    stock = operands.stock_routed(hidden, ids, weights)
    assert torch.equal(output.view(torch.int16), stock.view(torch.int16)), (
        "ring engine output differs bitwise from independent TRTLLM-Gen routed oracle"
    )
    return output


def _repeated_graph(state, hidden, layer, expected, repeats=64):
    output = torch.empty_like(expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        state.run(layer, hidden, out=output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state.run(layer, hidden, out=output)
    for _ in range(repeats):
        # Poison the output before every replay so a missing store cannot pass.
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))


@SM107
@pytest.mark.parametrize("rows", range(1, 9), ids=lambda n: f"m{n}")
@pytest.mark.parametrize("lag", [2, 3], ids=lambda n: f"lag{n}")
@pytest.mark.parametrize("geometry", ["shared24", "shared20", "routed"])
def test_tp8_ring_oracle_and_repeated_graph(runtime, monkeypatch, rows, lag, geometry):
    moe, provider, operands = runtime
    monkeypatch.setattr(moe, "M3_MOE_LAG", lag)
    shared_ctas = {"shared24": 24, "shared20": 20, "routed": 0}[geometry]
    # Set the actual host/kernel split together, including small-row SH20 cases.
    monkeypatch.setattr(moe, "shared_ctas_for", lambda m, shared: shared_ctas if shared else 0)
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert sms >= 24 + moe.MIN_CTAS, "test requires a valid resident-CTA split"
    state = moe.M3MoeState(device, rows, num_ctas=sms - shared_ctas, shared=bool(shared_ctas))
    assert int(state.mod.LAG) == lag
    assert int(state.mod.FC1_DONE_SLOTS) == (4 if rows <= 4 else 8)
    assert int(state.mod.NSHC) == shared_ctas
    assert not state.config.get("lag_unsafe"), "the default lag must not bypass admission"
    # All three routing distributions use a zero shared expert for stock parity.
    print(
        f"RING_CONFIG rows={rows} lag={lag} routed_ctas={state.num_ctas} shared_ctas={shared_ctas} "
        f"slots={int(state.mod.FC1_DONE_SLOTS)} stages={int(state.mod.STAGES)}",
        flush=True,
    )
    if rows == 8 and state.num_ctas == 188 and lag == 2:
        assert torch.count_nonzero(state.plan[29, 2, 2:8]).item() == 0
    for case in ("structured", "random", "concentrated"):
        print(f"RING_CASE_BEGIN rows={rows} lag={lag} geometry={geometry} case={case}", flush=True)
        layer = operands.layer_for(state, case, shared=False)
        if case == "concentrated":
            layer.bias.zero_()
            layer.bias[list(operands.CONCENTRATED)] = 2.0
        if case == "structured":
            hidden, expected_ids = _structured_inputs(operands, rows, layer)
        else:
            hidden, expected_ids = operands.hidden_states(rows, seed=300 + rows), None
        reference = _assert_routed_oracle(state, provider, operands, hidden, layer, expected_ids)
        if case == "concentrated":
            actual = state.route_dbg[:, :4].long().sort(dim=-1).values
            expected = torch.tensor(sorted(operands.CONCENTRATED), device="cuda").expand(rows, -1)
            assert torch.equal(actual, expected)
        _repeated_graph(state, hidden, layer, reference)
    # Nonzero shared-expert execution, changing inputs between captured replays.
    layer = operands.layer_for(state, "random", shared=True)
    hidden = operands.hidden_states(rows, seed=600 + rows)
    output = torch.empty_like(hidden)
    state.run(layer, hidden, out=output)  # compile/warm before capture
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        state.run(layer, hidden, out=output)
    for seed in (701, 702, 703):
        hidden.copy_(operands.hidden_states(rows, seed=seed))
        expected = state.run(layer, hidden).clone()
        _repeated_graph(state, hidden, layer, expected, repeats=4)
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
    print(f"RING_CASE_PASS rows={rows} lag={lag} geometry={geometry}", flush=True)
