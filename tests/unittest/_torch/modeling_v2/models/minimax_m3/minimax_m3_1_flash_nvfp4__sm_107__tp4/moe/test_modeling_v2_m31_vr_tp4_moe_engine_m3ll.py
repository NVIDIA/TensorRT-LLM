# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP4_MOE_ENGINE (default on in the sm_107 tp4 target): the reference fused MoE engine (vendored in the
family module minimax_m3/kernels/m3ll) built on the target's m3_moe state is m3_moe bit for bit.

* routing (route_dbg), ``out`` and the pushed set-A words equal m3_moe's at rows 1..8, on the served route pool
  (served 8-row route sets of a capture, forced through the fused router) and under a 2^+-12 activation block-scale
  stress and a wide weight block-scale stress;
* the 8-row B2LAG build (lag 6 on the 8-deep fc1_done ring) never hangs on the served route sets with the most
  distinct experts (>= 21: >= 6 groups, the heavy_fc2_only CTAs without FC1 work: the idle-CTA hang shape on m3_moe's
  4-slot ring), and a CUDA-graph replay gives the eager bits;
"""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.kernels.m3ll import (  # noqa: E501
    tp4_moe_engine as wc,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.m3_moe import (  # noqa: E501
    op as moe_op,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.m3_moe_engine import m3_moe_engine

__extra_import_path__ = ["../../../..", "../../ops/moe"]
from _m31_flash_moe_operands import (  # noqa: E402
    FC2_INPUT_SCALE,
    NUM_EXPERTS,
    SWIGLU_ALPHA,
    SWIGLU_LIMIT,
    experts,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)

H = 6144
I_TP = moe_op.INTER
EMPTY = -(2**31)
ROUTES = json.loads((Path(__file__).resolve().parent / "_m31_tp4wc_routes.json").read_text())[
    "routes"
]
HANG_DEADLINE_S = 120.0

_OURS: dict = {}
_HIS: dict = {}


def _ours(m: int):
    if m not in _OURS:
        dev = torch.device("cuda")
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        # the engine's base state, as the target builds it for the reference engine: no MOE_HANDOFF tables
        _OURS[m] = moe_op.M3MoeState(
            dev, num_tokens=m, num_ctas=sms - 24, shared=True, handoff=False
        )
    return _OURS[m]


def _reference(m: int, b2lag: bool = True, early_trig: bool = True):
    key = (m, b2lag, early_trig)
    if key not in _HIS:
        _HIS[key] = wc.build_state(
            torch.device("cuda"), _ours(m), b2lag=b2lag, early_trig=early_trig
        )
    return _HIS[key]


def _wide_operands(seed: int):
    """experts()'s layer with the routed experts' block scales over the whole normal E4M3 range (2^-6 .. 2^8) and the
    per-expert alphas rescaled to keep the sums finite: the weight-side block-scale stress."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import NVFP4TRTLLMGenFusedMoEMethod

    base = experts(I_TP)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    e = NUM_EXPERTS
    w31 = torch.randint(
        0, 256, (e, 2 * I_TP, H // 2), dtype=torch.uint8, device="cuda", generator=gen
    )
    w2 = torch.randint(0, 256, (e, H, I_TP // 2), dtype=torch.uint8, device="cuda", generator=gen)
    w31s = torch.randint(
        0x08, 0x78, (e, 2 * I_TP, H // 16), dtype=torch.uint8, device="cuda", generator=gen
    )
    w2s = torch.randint(
        0x08, 0x78, (e, H, I_TP // 16), dtype=torch.uint8, device="cuda", generator=gen
    )
    module = SimpleNamespace(
        w3_w1_weight=w31,
        w2_weight=w2,
        w3_w1_weight_scale=w31s.view(torch.float8_e4m3fn),
        w2_weight_scale=w2s.view(torch.float8_e4m3fn),
        is_gated_activation=True,
        bias=False,
    )
    NVFP4TRTLLMGenFusedMoEMethod()._shuffle_all_experts(module)
    torch.cuda.synchronize()
    fc31_alpha = base.tensors[4] / 64.0
    fc2_alpha = base.tensors[9] / 64.0
    tensors = (
        module.w3_w1_weight,
        module.w3_w1_weight_scale,
        module.w2_weight,
        module.w2_weight_scale,
        fc31_alpha,
        FC2_INPUT_SCALE * fc31_alpha,
        torch.full((e,), SWIGLU_ALPHA, device="cuda"),
        1.0 / fc31_alpha,
        SWIGLU_LIMIT / fc31_alpha,
        fc2_alpha,
    )
    return SimpleNamespace(
        tensors=tensors, shared=base.shared, fc31_input_scale=base.fc31_input_scale
    )


_WIDE = {}


def _operands(wide: bool):
    if not wide:
        return experts(I_TP)
    if "w" not in _WIDE:
        _WIDE["w"] = _wide_operands(20261101)
    return _WIDE["w"]


def _forced_router(ids):
    """Router weight / bias / hidden rows 0..7 that make the fused router pick ``ids`` [M][4] (in that order): token t
    has 1.0 at hidden dim t and 0 at dims 0..7 otherwise; the router reads only dims 0..7, expert ids[t][k] at 8 - k,
    every other expert at -8 (sigmoid gaps far above any rounding)."""
    m = len(ids)
    w = torch.zeros(NUM_EXPERTS, H, dtype=torch.float32, device="cuda")
    w[:, :m] = -8.0
    for t, row in enumerate(ids):
        for k, e in enumerate(row):
            w[e, t] = 8.0 - k
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    return w.contiguous(), bias


def _hidden(m: int, seed: int, stress: bool):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=gen)
    if stress:
        # 2^k on every 16-element activation block past the routing dims, k uniform in [-12, 12]
        k = torch.randint(-12, 13, (m, H // 16), device="cuda", generator=gen).float()
        k[:, 0] = 0
        x = x * torch.exp2(k).repeat_interleave(16, dim=1)
    x[:, :8] = 0.0
    x[torch.arange(m), torch.arange(m)] = 1.0
    return x.to(torch.bfloat16).contiguous()


def _layer(state, ids, wide: bool, builder=None):
    ops = _operands(wide)
    w, bias = _forced_router(ids)
    make = builder or state.layer
    return make(
        *ops.tensors,
        bias,
        router_weight=w,
        fc31_input_scale=ops.fc31_input_scale,
        shared_weights=ops.shared,
    )


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def _pushed(engine, hidden, layer, m):
    dev = torch.device("cuda")
    buf = torch.full((2 * 2 * 8 * H // 2,), EMPTY, dtype=torch.int32, device=dev)
    flags = torch.zeros(8 * 8, dtype=torch.int32, device=dev)
    arp = moe_op.ar_push_desc(buf.data_ptr(), flags.data_ptr(), 0, 1, 8, dev)
    engine(hidden, layer, None, arp)
    torch.cuda.synchronize()
    return buf[: m * H // 2].clone(), buf[m * H // 2 : 8 * H // 2].clone()


def _check_pair(
    m: int, ids, seed: int, wide: bool, stress: bool, b2lag: bool = True, early_trig: bool = True
) -> None:
    ours, ref = _ours(m), _reference(m, b2lag, early_trig)
    lo, lh = (
        _layer(ours, ids, wide),
        _layer(ref, ids, wide, builder=lambda *a, **k: wc.layer(ref, *a, **k)),
    )
    hidden = _hidden(m, seed, stress)
    out_o = m3_moe_engine(hidden, lo).clone()
    rd_o = ours.route_dbg.clone()
    out_h = m3_moe_engine(hidden, lh).clone()
    rd_h = ref.route_dbg.clone()
    torch.cuda.synchronize()
    assert torch.equal(rd_o, rd_h), f"m={m}: routing differs"
    assert torch.equal(
        rd_o[:, :4].long().sort(dim=1).values, torch.tensor(ids, device="cuda").sort(dim=1).values
    )
    assert _same(out_o, out_h), f"m={m} seed={seed}: out differs from m3_moe"
    po, ro = _pushed(m3_moe_engine, hidden, lo, m)
    ph, rh = _pushed(m3_moe_engine, hidden, lh, m)
    assert torch.equal(po, ph), f"m={m}: pushed words differ from m3_moe"
    assert bool((rh == EMPTY).all()) and bool((ro == EMPTY).all()), "a push beyond the rows"


def _route_rows(r, m):
    return r["ids"][:m]


@pytest.mark.parametrize("m", [1, 5, 8])
def test_bitwise_vs_m3_moe_on_served_routes(m: int) -> None:
    for j, r in enumerate(ROUTES[:: max(1, len(ROUTES) // 6)]):
        _check_pair(m, _route_rows(r, m), seed=1000 * m + j, wide=False, stress=False)


@pytest.mark.parametrize("m", (1, 8))
@pytest.mark.parametrize("wide", (False, True))
def test_bitwise_block_scale_stress(m: int, wide: bool) -> None:
    """2^+-12 activation block scales (the NVFP4 activation's block scales saturate and underflow) and, ``wide``,
    weight block scales over the whole normal E4M3 range: still m3_moe's bits."""
    for j, r in enumerate(ROUTES[:4] + ROUTES[-2:]):
        _check_pair(m, _route_rows(r, m), seed=9000 + 31 * m + j, wide=wide, stress=True)


def test_b2lag_ring_never_hangs_on_the_widest_served_routes() -> None:
    """lag 6 with >= 6 groups (>= 21 distinct experts) and m3_moe's heavy_fc2_only CTAs (no FC1 work at 8 rows): the
    idle-CTA hang shape. 64 calls per route set, the whole run under a deadline; a hang exits the process (rc 86)."""
    ref = _reference(8)
    assert moe_op.HEAVY_FC2_ONLY_BY_M[7], "the 8-row plan has CTAs without FC1 work"
    wide = [r for r in ROUTES if r["E"] >= 21]
    assert wide, "the fixture holds >= 21-expert route sets"
    calls = 0
    done = torch.cuda.Event()
    for r in wide:
        layer = _layer(ref, r["ids"], False, builder=lambda *a, **k: wc.layer(ref, *a, **k))
        hidden = _hidden(8, seed=r["step"], stress=False)
        for _ in range(64):
            m3_moe_engine(hidden, layer)
            calls += 1
    done.record()
    t0 = time.monotonic()
    while not done.query():
        if time.monotonic() - t0 > HANG_DEADLINE_S:
            print(
                f"HANG: tp4 moe engine did not finish {calls} calls in {HANG_DEADLINE_S} s",
                flush=True,
            )
            os._exit(86)
        time.sleep(0.05)
    assert calls == 64 * len(wide)


def test_cuda_graph_replay_is_eager_bits() -> None:
    ours, ref = _ours(8), _reference(8)
    r = ROUTES[-1]
    lo = _layer(ours, r["ids"], False)
    lh = _layer(ref, r["ids"], False, builder=lambda *a, **k: wc.layer(ref, *a, **k))
    static = _hidden(8, seed=5, stress=False)
    out = torch.empty(8, H, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        m3_moe_engine(static, lh, out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        m3_moe_engine(static, lh, out)
    for seed in (6, 7, 8):
        static.copy_(_hidden(8, seed=seed, stress=False))
        graph.replay()
        torch.cuda.synchronize()
        want = m3_moe_engine(static, lo).clone()
        torch.cuda.synchronize()
        assert _same(out, want), f"seed {seed}: replay differs from m3_moe"
