# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Oracle for #117, the sparse QKV + index GEMV with the producer in its owner epilogue (mx_gemv
``m3_mx_gemv_producer``): q, index-q and every cache byte (NVFP4 K / V data and scales, FP8 index-K) bit for bit equal
to today's chain, the GEMV storing the packed BF16 row and then the producer kernel on it. Both producer builds are
references: the library's ``minimax_m3_producer_pw`` and, when the target's extension is loaded, its own
``minimax_m3_sm107_tp8_producer_pw`` (the copy the target dispatches, with its pre-wait step-input reads)."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp8.ops.mx_gemv import (  # noqa: E501
    op as gemv,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

N, K = 12 * 128, 6144
HEADS_Q = 8
PAGE = 128
PAGES = 512
MAX_POSITIONS = 262144
EPS = 1e-6
KV_QUANT = (1.0, 0.37, 0.61)


def _case(tokens, seed, scale=1.0, bad_slots=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device="cuda") * (scale / K**0.5)).to(torch.bfloat16)
    x = torch.randn(tokens, K, generator=g, device="cuda").to(torch.bfloat16)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    rotary = torch.randn(MAX_POSITIONS, 2, 32, device="cuda", generator=g)
    norms = [(torch.randn(128, device="cuda", generator=g) * 0.3).bfloat16() for _ in range(4)]
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=g)[:tokens].int()
    if bad_slots and tokens >= 3:
        slots[1] = -1
        slots[2] = PAGES * PAGE + 5
    positions = torch.randint(
        0, MAX_POSITIONS, (tokens,), device="cuda", dtype=torch.int32, generator=g
    )
    quant = torch.tensor(KV_QUANT, device="cuda")
    return xq.view(tokens, K), xsf, wq.view(N, K), wsf, slots, quant, norms, rotary, positions


def _caches():
    return (
        torch.zeros(PAGES, 2, 1, PAGE, 64, dtype=torch.uint8, device="cuda"),
        torch.zeros(PAGES, 2, 1, PAGE, 8, dtype=torch.uint8, device="cuda"),
        torch.zeros(PAGES, 1, PAGE, 128, dtype=torch.float8_e4m3fn, device="cuda"),
    )


def _chain(case, caches, producer):
    xq, xsf, wq, wsf, slots, quant, norms, rotary, positions = case
    packed = gemv.m3_mx_gemv(xq, xsf, wq, wsf, use_pdl=True)
    assert packed is not None, "the sparse QKV shape must be served"
    if producer == "lib":
        q, iq = torch.ops.trtllm.minimax_m3_producer_pw(
            packed, *caches, slots, quant, EPS, *norms, rotary, positions, HEADS_Q, True, True
        )
    else:
        q, iq = torch.ops.trtllm.minimax_m3_sm107_tp8_producer_pw(
            packed, *caches, slots, quant, EPS, *norms, rotary, positions, True, True, True
        )
    torch.cuda.synchronize()
    return q, iq


def _fused(case, caches):
    xq, xsf, wq, wsf, slots, quant, norms, rotary, positions = case
    tokens = xq.shape[0]
    q = torch.zeros(tokens, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.zeros(tokens, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
    kv_data, kv_scale, index_k = caches
    ok = gemv.m3_mx_gemv_producer(
        xq, xsf, wq, wsf, q, iq, kv_data, kv_scale, index_k, slots, quant, EPS, *norms, rotary, positions,
        use_pdl=True,
    )  # fmt: skip
    assert ok, "the producer GEMV must serve the sparse QKV shape at <= 8 tokens"
    torch.cuda.synchronize()
    return q, iq


def _producers():
    out = ["lib"]
    if hasattr(torch.ops.trtllm, "minimax_m3_sm107_tp8_producer_pw"):
        out.append("tp8")
    return out


@pytest.mark.parametrize(
    "tokens, seed, scale, bad_slots",
    [
        (1, 21, 1.0, False),
        (2, 22, 1.0, False),
        (4, 23, 1.0, False),
        (4, 24, 40.0, False),
        (8, 25, 1.0, False),
        (8, 26, 0.01, False),
        (8, 27, 1.0, True),
    ],
    ids=["t1", "t2", "t4", "t4-large", "t8", "t8-tiny", "t8-bad-slots"],
)
@pytest.mark.parametrize("producer", _producers())
def test_producer_gemv_is_bitwise_the_chain(tokens, seed, scale, bad_slots, producer) -> None:
    case = _case(tokens, seed, scale, bad_slots)
    ref_caches = _caches()
    rq, riq = _chain(case, ref_caches, producer)
    caches = _caches()
    q, iq = _fused(case, caches)
    names = ("q", "index_q", "kv_data", "kv_scale", "index_k")
    for name, a in zip(names, (rq, riq, *ref_caches)):
        assert bool(a.contiguous().view(torch.uint8).any()), f"reference {name} is all zero"
    for name, a, b in zip(names, (rq, riq, *ref_caches), (q, iq, *caches)):
        a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        diff = (a8 != b8).nonzero()
        assert diff.numel() == 0, (
            f"{name}: {diff.shape[0]} bytes differ, first at {tuple(diff[0].tolist())}"
        )


def test_producer_gemv_is_deterministic_and_graph_replays() -> None:
    """Three eager runs and a captured replay write the same bytes."""
    case = _case(4, 31)
    outs = []
    for _ in range(3):
        caches = _caches()
        q, iq = _fused(case, caches)
        outs.append([t.contiguous().view(torch.uint8).clone() for t in (q, iq, *caches)])
    xq, xsf, wq, wsf, slots, quant, norms, rotary, positions = case
    q = torch.zeros(4, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.zeros(4, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
    caches = _caches()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        gemv.m3_mx_gemv_producer(
            xq, xsf, wq, wsf, q, iq, *caches, slots, quant, EPS, *norms, rotary, positions, use_pdl=True
        )  # fmt: skip
    graph.replay()
    torch.cuda.synchronize()
    outs.append([t.contiguous().view(torch.uint8).clone() for t in (q, iq, *caches)])
    for run in outs[1:]:
        for a, b in zip(outs[0], run):
            assert torch.equal(a, b)


def test_declines_outside_its_shape() -> None:
    """More than 8 tokens, or a projection other than the 12-head packed one, launches nothing."""
    case = _case(8, 41)
    xq, xsf, wq, wsf, slots, quant, norms, rotary, positions = case
    wide = torch.cat([xq, xq], dim=0)
    q = torch.zeros(16, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.zeros(16, 1, 128, dtype=torch.float8_e4m3fn, device="cuda")
    xsf16 = torch.ops.trtllm.mxfp8_quantize(
        torch.zeros(16, K, dtype=torch.bfloat16, device="cuda"), True
    )[1]
    caches = _caches()
    assert not gemv.m3_mx_gemv_producer(
        wide, xsf16, wq, wsf, q, iq, *caches, slots, quant, EPS, *norms, rotary, positions
    )  # fmt: skip
