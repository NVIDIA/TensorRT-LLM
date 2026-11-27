# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the decode MoE takes the route front (`DP_ROUTE_FRONT_ENV`).

* The tactic a gathered row count pins: `_ROUTE_FRONT_TACTICS`' where the runner lists it for
  the window, else the base table's, else none (that row count keeps the runner's routing).
* `_routed_experts` takes the route front only on a captured step's gathered rows with the
  decode router and no mixer-side activation, and only at a row count the build compiled.
* The build is off with the switch at 0, away from attention DP, on host weights, and away
  from the certified window (128 experts, dep4) and GPU (SM103).

The ops are recorders; no kernel runs.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

CORE = SimpleNamespace(topk=10, hidden=2560, expert_inter=640, local_experts=128)


class _Runner:
    def __init__(self, valid):
        self.valid = valid

    def get_valid_configs(self, top_k, hidden, inter, num_experts, rows):
        assert (top_k, hidden, inter, num_experts) == (10, 2560, 640, 128)
        return [list(t) for t in self.valid]


@pytest.mark.parametrize(
    "rows, valid, expected",
    [
        (48, [(8, 53)], (8, 53)),
        (96, [(16, 32), (8, 53)], (16, 32)),
        (192, [(32, 32)], (32, 32)),
        (704, [(64, 4), (16, 32)], (64, 4)),
        (704, [(32, 20)], (32, 20)),  # not listed: the base table's tactic at 704 rows
        (1024, [(32, 20)], (32, 20)),
        (1024, [(8, 0)], None),
    ],
)
def test_pinned_tactic(rows, valid, expected):
    assert M._route_front_tactic(_Runner(valid), CORE, rows) == expected


def _engage(monkeypatch, capturing: bool, built_rows=(48,)):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    monkeypatch.setattr(
        M, "_route_front_run", lambda core, st, x, L, layer, logits: (calls.append("front"), 0)
    )
    monkeypatch.setattr(M, "_decode_router", lambda x, router: "logits")

    def runner(logits, *args):
        calls.append("runner")
        return torch.zeros(1)

    monkeypatch.setattr(M, "fp4_block_scale_moe_renorm_routed", runner)
    monkeypatch.setattr(M, "fp4_quantize", lambda *a: (torch.zeros(1), torch.zeros(1)))
    st = M._RouteFront(None)
    st.tactics = {rows: (8, 53) for rows in built_rows}
    L = SimpleNamespace(fc1_w=torch.zeros(1), router=None, moe_gscale=None, fc1_w_cd=None)
    st.layers = {L.fc1_w.data_ptr(): 0}
    for name in ("fc1_s", "fc2_w", "fc2_s", "out1_scale", "out1_gate_scale", "out2_scale"):
        setattr(L, name, None)
    core = SimpleNamespace(
        _route_front=st,
        trunk_moe_backend="TRTLLM",
        one_rank=False,
        hidden=8,
        num_experts=512,
        topk=10,
        expert_inter=640,
        expert_offset=0,
        local_experts=128,
    )
    return core, L, calls


@pytest.mark.parametrize(
    "capturing, router, fp4, rows, expected",
    [
        (True, "decode", None, 48, ["front"]),
        (False, "decode", None, 48, ["runner"]),  # an eager step
        (True, "prefill", None, 48, ["runner"]),  # another router
        (True, "decode", "mixer", 48, ["runner"]),  # the mixer's activation (attention TP)
        (True, "decode", None, 64, ["runner"]),  # a row count the build did not compile
    ],
)
def test_routed_experts_takes_the_route_front(monkeypatch, capturing, router, fp4, rows, expected):
    core, L, calls = _engage(monkeypatch, capturing)
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    router_mm = M._decode_router if router == "decode" else M.cublas_mm
    if router == "prefill":
        monkeypatch.setattr(M, "cublas_mm", lambda x, w: "logits")
        router_mm = M.cublas_mm
    act = None if fp4 is None else (torch.zeros(rows, 4), torch.zeros(rows))
    M._routed_experts(core, x, L, act, router_mm)
    assert calls == expected


def test_build_is_off_with_the_switch(monkeypatch):
    monkeypatch.setenv(M.DP_ROUTE_FRONT_ENV, "0")
    assert M._route_front_build(SimpleNamespace(), [], torch.device("cuda")) is None


@pytest.mark.parametrize(
    "dp_size, backend, device",
    [(1, "TRTLLM", "cuda"), (4, "CUTEDSL", "cuda"), (4, "TRTLLM", "cpu")],
)
def test_build_declines_away_from_attention_dp(monkeypatch, dp_size, backend, device):
    monkeypatch.delenv(M.DP_ROUTE_FRONT_ENV, raising=False)
    core = SimpleNamespace(
        dp_size=dp_size,
        one_rank=False,
        trunk_moe_backend=backend,
        num_experts=512,
        topk=10,
        local_experts=128,
    )
    assert M._route_front_build(core, [], torch.device(device)) is None


@pytest.mark.parametrize(
    "local_experts, capability",
    [(64, (10, 3)), (256, (10, 3)), (128, (10, 7))],
    ids=["dep8_window", "dep2_window", "sm_107"],
)
def test_build_declines_away_from_the_certified_window_and_arch(
    monkeypatch, local_experts, capability
):
    """The tactic table and the op's cells cover the dep4 window on SM103 only."""
    monkeypatch.delenv(M.DP_ROUTE_FRONT_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    core = SimpleNamespace(
        dp_size=4,
        one_rank=False,
        trunk_moe_backend="TRTLLM",
        num_experts=512,
        topk=10,
        local_experts=local_experts,
    )
    assert M._route_front_build(core, [], torch.device("cuda")) is None
