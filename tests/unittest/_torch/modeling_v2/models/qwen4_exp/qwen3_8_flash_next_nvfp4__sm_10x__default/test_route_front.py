# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the decode MoE takes the route front (`DP_ROUTE_FRONT_ENV`, and at a tep4 rank
`TEP_FAST_DECODE_ENV`).

* The tactic a gathered row count pins: `_ROUTE_FRONT_TACTICS`' where the runner lists it for
  the window, else the base table's, else none (that row count keeps the runner's routing). A
  tep4 rank pins the runner's own bucket tactic (`DECODE_TACTICS`). dep8's 64-expert window pins
  `_ROUTE_FRONT_TACTICS_W64`'s, with a listed reference where the runner does not list it, which
  the build check runs the runner's own routing at.
* `_routed_experts` takes the route front only on a captured step with the decode router, at a
  row count the build compiled: under attention DP on the gathered rows without a mixer-side
  activation, at a tep4 rank on its own rows with the site's activation, which the runner reads.
* The build is off with the switch at 0, away from attention DP and tep4, on host weights, and
  away from the certified windows (128 experts at dep4 / tep4, 64 at dep8) and GPU (SM103).

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
    def __init__(self, valid, window=128):
        self.valid, self.window = valid, window

    def get_valid_configs(self, top_k, hidden, inter, num_experts, rows):
        assert (top_k, hidden, inter, num_experts) == (10, 2560, 640, self.window)
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


W64 = SimpleNamespace(topk=10, hidden=2560, expert_inter=640, local_experts=64)


@pytest.mark.parametrize(
    "rows, valid, expected",
    [
        (64, [(8, 53), (16, 32)], ((8, 53), None)),  # 2 requests a rank, listed
        (96, [(16, 32)], ((16, 32), None)),
        # 14 requests a rank: tile 32 is not listed at 448 rows; held to the listed same tile
        (448, [(64, 4), (32, 20)], ((32, 32), (32, 20))),
        # 24 a rank (768 rows): no tile 32 listed, held to the first listed config
        (768, [(64, 4), (128, 4)], ((32, 32), (64, 4))),
        (896, [(64, 5), (128, 4)], ((64, 5), None)),
        (1024, [(128, 4)], ((64, 5), (128, 4))),
        (1024, [], (None, None)),
    ],
)
def test_dep8_pinned_tactic(rows, valid, expected):
    assert M._route_front_w64_tactic(_Runner(valid, 64), W64, rows) == expected


@pytest.mark.parametrize("ref", [None, (32, 20)])
def test_check_runs_the_runners_routing_at_the_listed_reference(monkeypatch, ref):
    """The build check's stock window call runs at the listed reference where the pinned tactic
    is not listed, else at the pinned tactic itself."""
    seen = []

    class Runner:
        def run_moe(self, *args):
            seen.append(tuple(args[26]))
            return (torch.zeros(2, 8),)

    monkeypatch.setattr(M, "cublas_mm", lambda x, w: torch.zeros(x.shape[0], 512))
    monkeypatch.setattr(M, "fp4_quantize", lambda *a: (torch.zeros(1), torch.zeros(1)))
    monkeypatch.setattr(
        M, "renorm_moe_routing_op", lambda *a, **k: (torch.zeros(2, 10), torch.zeros(2, 10))
    )
    out = (
        torch.zeros(2, 8),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(2, 10),
        torch.zeros(2, 10),
    )
    monkeypatch.setattr(M, "_route_front_run", lambda *a, **k: out)
    monkeypatch.setattr(M, "_route_front_check_tables", lambda *a: "")
    st = M._RouteFront(Runner())
    st.tactics = {2: (32, 32)}
    if ref is not None:
        st.refs = {2: ref}
    st.views = {2: [None]}
    L = SimpleNamespace(router=torch.zeros(8, 512), moe_gscale=None)
    for name in ("fc1_w", "fc1_s", "fc2_w", "fc2_s", "out1_scale", "out1_gate_scale", "out2_scale"):
        setattr(L, name, None)
    core = SimpleNamespace(
        hidden=8, num_experts=512, topk=10, expert_inter=640, expert_offset=0, local_experts=64
    )
    assert M._route_front_check(core, st, L, 0, 2, torch.Generator()) == ""
    assert seen == [ref or (32, 32)]


def _engage(monkeypatch, capturing: bool, built_rows=(48,), dp_size=4):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    def front(core, st, x, L, layer, logits, fp4=None):
        calls.append("front" if fp4 is None else "front+site")
        return (0,)

    monkeypatch.setattr(M, "_route_front_run", front)
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
        dp_size=dp_size,
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


@pytest.mark.parametrize(
    "capturing, fp4, rows, expected",
    [
        (True, "mixer", 16, ["front+site"]),  # the runner reads the site's codes
        (True, None, 16, ["runner"]),  # tep4 rows always come with the site's codes
        (False, "mixer", 16, ["runner"]),  # an eager step
        (True, "mixer", 12, ["runner"]),  # a row count the build did not compile
    ],
)
def test_routed_experts_takes_the_route_front_at_tep4(monkeypatch, capturing, fp4, rows, expected):
    core, L, calls = _engage(monkeypatch, capturing, built_rows=(4, 16), dp_size=1)
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    act = None if fp4 is None else (torch.zeros(rows, 4), torch.zeros(rows))
    M._routed_experts(core, x, L, act, M._decode_router)
    assert calls == expected


@pytest.mark.parametrize(
    "rows, valid, expected",
    [
        (4, [(8, 44), (8, 0)], (8, 44)),  # bucket 4
        (8, [(8, 0)], (8, 0)),  # bucket 8
        (12, [(8, 0)], (8, 0)),  # bucket 8
        (16, [(8, 44)], (8, 44)),  # bucket 16
        (16, [(8, 0)], None),  # the bucket's pin is not listed for the window
    ],
)
def test_tep4_pinned_tactic_is_the_runners_bucket(rows, valid, expected):
    assert M._tep_route_front_tactic(_Runner(valid), CORE, rows) == expected


def test_build_is_off_with_the_switch(monkeypatch):
    monkeypatch.setenv(M.DP_ROUTE_FRONT_ENV, "0")
    core = SimpleNamespace(dp_size=4)
    assert M._route_front_build(core, [], torch.device("cuda")) is None


@pytest.mark.parametrize(
    "dp_size, backend, device",
    [(1, "TRTLLM", "cuda"), (4, "CUTEDSL", "cuda"), (4, "TRTLLM", "cpu")],
)
def test_build_declines_away_from_attention_dp(monkeypatch, dp_size, backend, device):
    monkeypatch.delenv(M.DP_ROUTE_FRONT_ENV, raising=False)
    core = SimpleNamespace(
        dp_size=dp_size,
        tep_fast=False,
        one_rank=False,
        trunk_moe_backend=backend,
        num_experts=512,
        topk=10,
        local_experts=128,
    )
    assert M._route_front_build(core, [], torch.device(device)) is None


@pytest.mark.parametrize(
    "dp_size, local_experts, capability",
    [
        (4, 64, (10, 3)),
        (8, 128, (10, 3)),
        (2, 256, (10, 3)),
        (4, 128, (10, 7)),
        (8, 64, (10, 7)),
        (1, 128, (10, 7)),
    ],
    ids=["dep4_64", "dep8_128", "dep2_window", "sm_107", "dep8_sm_107", "tep4_sm_107"],
)
def test_build_declines_away_from_the_certified_window_and_arch(
    monkeypatch, dp_size, local_experts, capability
):
    """The tactic tables and the op's cells cover the dep4 / tep4 (128 experts) and dep8 (64)
    windows on SM103 only."""
    monkeypatch.delenv(M.DP_ROUTE_FRONT_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    core = SimpleNamespace(
        dp_size=dp_size,
        tep_fast=dp_size == 1,
        one_rank=False,
        trunk_moe_backend="TRTLLM",
        num_experts=512,
        topk=10,
        local_experts=local_experts,
    )
    assert M._route_front_build(core, [], torch.device("cuda")) is None


class _Reached(Exception):
    pass


@pytest.mark.parametrize("dp_size, local_experts", [(4, 128), (8, 64)], ids=["dep4", "dep8"])
def test_build_takes_the_certified_windows(monkeypatch, dp_size, local_experts):
    """dep4's 128-expert and dep8's 64-expert windows on SM103 reach the runner's tables."""
    monkeypatch.delenv(M.DP_ROUTE_FRONT_ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 3))

    def runner():
        raise _Reached

    monkeypatch.setattr(M, "_moe_runner", runner)
    core = SimpleNamespace(
        dp_size=dp_size,
        tep_fast=False,
        one_rank=False,
        trunk_moe_backend="TRTLLM",
        num_experts=512,
        topk=10,
        local_experts=local_experts,
        max_verify_rows=4,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=[3, 12]),
    )
    with pytest.raises(_Reached):
        M._route_front_build(core, [], torch.device("cuda"))
