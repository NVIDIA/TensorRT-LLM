# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QA of the tp8 m3_moe engine at the experimental phase lags 5 and 6, which the served engine never admits
(m31_moe_lag.phase_lag_config refuses them unless asked with ``allow_unsafe=True``; this test asks for it on the
engine module only).

QA only, not in a CI list: run under a process timeout on SM107 with TRTLLM_M31_RUN_HIGH_LAG_QA=1. Uses the ring test's
stock-routed oracle and graph helpers. The nonzero shared-expert section checks repeatability, not independent
shared-expert math.
"""

import functools
import os

import pytest
import torch
from test_m31_tp8_ring_gpu import SM107, _assert_routed_oracle, _repeated_graph, _structured_inputs
from test_m31_tp8_ring_gpu import runtime as _ring_runtime

# Register the shared fixture here even when the normal ring module was collected first.
runtime = _ring_runtime


@SM107
@pytest.mark.skipif(
    os.environ.get("TRTLLM_M31_RUN_HIGH_LAG_QA") != "1",
    reason="experimental lag5/6 QA requires explicit opt-in",
)
@pytest.mark.parametrize("rows", range(1, 9), ids=lambda n: f"m{n}")
@pytest.mark.parametrize("lag", [5, 6], ids=lambda n: f"lag{n}")
@pytest.mark.parametrize("geometry", ["shared24", "shared20", "routed"])
def test_tp8_high_lag_oracle_and_repeated_graph(runtime, monkeypatch, rows, lag, geometry):
    moe, provider, operands = runtime
    monkeypatch.setattr(
        moe, "phase_lag_config", functools.partial(moe.phase_lag_config, allow_unsafe=True)
    )
    monkeypatch.setattr(moe, "M3_MOE_LAG", lag)
    shared_ctas = {"shared24": 24, "shared20": 20, "routed": 0}[geometry]
    monkeypatch.setattr(moe, "shared_ctas_for", lambda m, shared: shared_ctas if shared else 0)
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    assert sms >= 24 + moe.MIN_CTAS
    state = moe.M3MoeState(device, rows, num_ctas=sms - shared_ctas, shared=bool(shared_ctas))
    assert int(state.mod.LAG) == lag
    assert int(state.mod.FC1_DONE_SLOTS) == (4 if rows <= 4 else 8)
    assert int(state.mod.NSHC) == shared_ctas
    assert state.config.get("lag_unsafe") == 1, "experimental admission was not explicit"
    print(
        f"HIGH_LAG_CONFIG rows={rows} lag={lag} routed_ctas={state.num_ctas} "
        f"shared_ctas={shared_ctas} slots={int(state.mod.FC1_DONE_SLOTS)} "
        f"stages={int(state.mod.STAGES)} unsafe={state.config['lag_unsafe']}",
        flush=True,
    )
    for case in ("structured", "random", "concentrated"):
        print(
            f"HIGH_LAG_CASE_BEGIN rows={rows} lag={lag} geometry={geometry} case={case}", flush=True
        )
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
    layer = operands.layer_for(state, "random", shared=True)
    hidden = operands.hidden_states(rows, seed=600 + rows)
    output = torch.empty_like(hidden)
    state.run(layer, hidden, out=output)
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
    print(f"HIGH_LAG_CASE_PASS rows={rows} lag={lag} geometry={geometry}", flush=True)
