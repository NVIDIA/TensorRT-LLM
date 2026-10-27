# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Oracle for QKV_PRODUCER_FOLD on the sm_107 TP2 target: the sparse QKV + index GEMV with the producer in its owner
epilogue (mx_gemv ``m3_mx_gemv_producer``) writes q, index-q and every cache byte (NVFP4 K / V data and scales of both
KV heads, FP8 index-K) bit for bit as today's chain: the in-model GEMV (split-8 override ``mx_gemv_cfg``) storing the
packed BF16 row, then the target's producer on it. Both producer builds are references: the target's own
``m3_producer`` (``minimax_m3_sm107_tp2_producer_pw``, with its pre-wait step-input reads) and the native
``minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert``."""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels.mx_gemv import (  # noqa: E501
    op as gemv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention.m3_producer import (  # noqa: E501
    m3_producer,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

N, K = 39 * 128, 6144
HEADS_Q, HEADS_KV, HEADS_INDEX = 32, 2, 2
PAGE = 128
PAGES = 512
MAX_POSITIONS = 262144
EPS = 1e-6
KV_QUANT = (1.0, 0.37, 0.61)
CFG = modeling.mx_gemv_cfg(N, K, 8)  # the in-model split-8 override


def _block_scaled(rows, cols, g, lo, hi, base_scale=1.0):
    """bf16 [rows, cols], every 32-element block scaled by 2^U{lo..hi} (block-scale stress when lo / hi are wide)."""
    base = torch.randn(rows, cols, device="cuda", generator=g) * base_scale
    e = torch.randint(lo, hi + 1, (rows, cols // 32), device="cuda", generator=g).float()
    return (base * torch.exp2(e).repeat_interleave(32, dim=1)).to(torch.bfloat16)


def _side(tokens, g, bad_slots=False):
    rotary = torch.randn(MAX_POSITIONS, 2, 32, device="cuda", generator=g)
    norms = [(torch.randn(128, device="cuda", generator=g) * 0.3).bfloat16() for _ in range(4)]
    slots = torch.randperm(PAGES * PAGE, device="cuda", generator=g)[:tokens].int()
    if bad_slots and tokens >= 3:
        slots[1] = -1
        slots[2] = PAGES * PAGE + 5
    positions = torch.randint(
        0, MAX_POSITIONS, (tokens,), device="cuda", dtype=torch.int32, generator=g
    )
    return slots, torch.tensor(KV_QUANT, device="cuda"), norms, rotary, positions


def _case(tokens, seed, scale=1.0, bad_slots=False, stress=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = (-12, 12) if stress else (0, 0)
    w = _block_scaled(N, K, g, lo, hi, scale / K**0.5)
    x = _block_scaled(tokens, K, g, lo, hi)
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w, True)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, True)
    return (xq.view(tokens, K), xsf, wq.view(N, K), wsf, *_side(tokens, g, bad_slots))


def _caches():
    return (
        torch.zeros(PAGES, 2, HEADS_KV, PAGE, 64, dtype=torch.uint8, device="cuda"),
        torch.zeros(PAGES, 2, HEADS_KV, PAGE, 8, dtype=torch.uint8, device="cuda"),
        torch.zeros(PAGES, 1, PAGE, 128, dtype=torch.float8_e4m3fn, device="cuda"),
    )


def _chain(case, caches, producer, eps=EPS):
    xq, xsf, wq, wsf, slots, quant, norms, rotary, positions = case
    packed = gemv.m3_mx_gemv(xq, xsf, wq, wsf, True, CFG, True, None)
    assert packed is not None, "the in-model sparse QKV GEMV must serve <= 8 tokens"
    if producer == "target":
        q, iq = m3_producer(
            packed, *caches, slots, quant, eps, *norms, rotary, positions, HEADS_Q, True, True, True
        )  # fmt: skip
    else:
        q, iq = torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
            packed, *caches, slots, quant, HEADS_Q, HEADS_KV, HEADS_INDEX, 128, 64, eps, *norms, rotary, positions
        )  # fmt: skip
    torch.cuda.synchronize()
    return q, iq


def _fused(case, caches, eps=EPS, cfg=CFG):
    xq, xsf, wq, wsf, slots, quant, norms, rotary, positions = case
    tokens = xq.shape[0]
    q = torch.zeros(tokens, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.zeros(tokens, HEADS_INDEX, 128, dtype=torch.float8_e4m3fn, device="cuda")
    ok = gemv.m3_mx_gemv_producer(
        xq, xsf, wq, wsf, q, iq, *caches, slots, quant, eps, *norms, rotary, positions, cfg=cfg, use_pdl=True
    )  # fmt: skip
    torch.cuda.synchronize()
    return ok, q, iq


def _assert_bytes_equal(tag, ref, out):
    names = ("q", "index_q", "kv_data", "kv_scale", "index_k")
    for name, a, b in zip(names, ref, out):
        a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        diff = (a8 != b8).nonzero()
        assert diff.numel() == 0, (
            f"{tag}: {name}: {diff.shape[0]} bytes differ, first at {tuple(diff[0].tolist())}"
        )


CASES = [
    (1, 21, 1.0, False, False),
    (2, 22, 1.0, False, False),
    (4, 23, 1.0, False, False),
    (4, 24, 40.0, False, False),
    (8, 25, 1.0, False, False),
    (8, 26, 0.01, False, False),
    (8, 27, 1.0, True, False),
    (1, 28, 1.0, False, True),
    (5, 29, 1.0, False, True),
    (8, 30, 1.0, True, True),
]


@pytest.mark.parametrize(
    "tokens, seed, scale, bad_slots, stress",
    CASES,
    ids=[
        "t1",
        "t2",
        "t4",
        "t4-large",
        "t8",
        "t8-tiny",
        "t8-bad-slots",
        "t1-stress",
        "t5-stress",
        "t8-stress-bad",
    ],
)
@pytest.mark.parametrize("producer", ["target", "native"])
def test_fold_is_bitwise_the_chain(tokens, seed, scale, bad_slots, stress, producer) -> None:
    """q, index-q (both heads) and every cache byte (both KV heads' NVFP4 data and scales, index-K), on natural and
    2^+-12 block-scale stress operands."""
    case = _case(tokens, seed, scale, bad_slots, stress)
    ref_caches = _caches()
    rq, riq = _chain(case, ref_caches, producer)
    caches = _caches()
    ok, q, iq = _fused(case, caches)
    assert ok, (
        "the producer GEMV must serve the sparse QKV shape at <= 8 tokens on the split-8 config"
    )
    for name, a in zip(("q", "index_q", "kv_data", "kv_scale", "index_k"), (rq, riq, *ref_caches)):
        assert bool(a.contiguous().view(torch.uint8).any()), f"reference {name} is all zero"
    # Both KV heads and both index-q heads carry data (the fold's head mapping, not only head 0).
    for h in range(HEADS_KV):
        assert bool(ref_caches[0][:, :, h].any()), f"reference KV head {h} is empty"
    for h in range(HEADS_INDEX):
        assert bool(riq[:, h].contiguous().view(torch.uint8).any()), (
            f"reference index-q head {h} is empty"
        )
    _assert_bytes_equal(f"{producer} t{tokens}", (rq, riq, *ref_caches), (q, iq, *caches))


def _edge_case(tokens, seed, edge):
    """The producer's arithmetic edges on the packed row: the GEMV weight is an identity on the first N inputs, so the
    packed BF16 row is the MXFP8-rounded input row."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = torch.randn(tokens, N // 128, 128, device="cuda", generator=g)
    if edge == "zero_rows":
        rows[0] = 0.0
        rows[1, HEADS_Q + 2] = 0.0  # a zero V row of KV head 0: an all-zero NVFP4 block
        rows[1, HEADS_Q + 3] = 0.0  # and of KV head 1
    elif edge == "denormal_rms":
        rows[:2] = (
            rows[:2] * 1e-20
        )  # mean square below FLT_MIN with eps = 0: the rsqrt's denormal path
    elif edge == "saturate":
        rows[0] = (
            rows[0] * 1e4
        )  # past E4M3 448 after the norm's (1 + w), and saturating NVFP4 block scales
        rows[1, :, ::7] = 3e38
    elif edge == "nan_inf":
        rows[0, :, 5] = float("nan")
        rows[1, :, 9] = float("inf")
        rows[2, :, 40] = -float("inf")
    x = torch.zeros(tokens, K, device="cuda")
    x[:, :N] = rows.reshape(tokens, N)
    w = torch.zeros(N, K, device="cuda")
    w[:, :N] = torch.eye(N, device="cuda")
    wq, wsf = torch.ops.trtllm.mxfp8_quantize(w.to(torch.bfloat16), True)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x.to(torch.bfloat16), True)
    slots, quant, norms, rotary, positions = _side(tokens, g)
    positions[0] = 0
    positions[-1] = MAX_POSITIONS - 1  # the RoPE table's first and last rows
    if edge == "rope_edges":
        positions[1] = 1
        positions[2] = MAX_POSITIONS - 2
    return xq.view(tokens, K), xsf, wq.view(N, K), wsf, slots, quant, norms, rotary, positions


EDGES = [
    ("zero_rows", 1e-6),
    ("denormal_rms", 0.0),
    ("saturate", 1e-6),
    ("nan_inf", 1e-6),
    ("rope_edges", 1e-6),
]


@pytest.mark.parametrize("edge, eps", EDGES, ids=[e[0] for e in EDGES])
def test_fold_edge_rows(edge, eps) -> None:
    """Zero rows, a denormal RMS argument (eps 0), E4M3 / NVFP4 saturation, NaN / infinity and the RoPE table's edge
    rows: the fold writes the chain's bytes (target producer)."""
    case = _edge_case(8, 50 + len(edge), edge)
    ref_caches = _caches()
    rq, riq = _chain(case, ref_caches, "target", eps)
    caches = _caches()
    ok, q, iq = _fused(case, caches, eps)
    assert ok
    _assert_bytes_equal(edge, (rq, riq, *ref_caches), (q, iq, *caches))


def test_fold_is_deterministic_and_graph_replays() -> None:
    """Three eager runs and a captured replay write the same bytes."""
    case = _case(4, 31)
    outs = []
    for _ in range(3):
        caches = _caches()
        _, q, iq = _fused(case, caches)
        outs.append([t.contiguous().view(torch.uint8).clone() for t in (q, iq, *caches)])
    xq, xsf, wq, wsf, slots, quant, norms, rotary, positions = case
    q = torch.zeros(4, HEADS_Q, 128, dtype=torch.float8_e4m3fn, device="cuda")
    iq = torch.zeros(4, HEADS_INDEX, 128, dtype=torch.float8_e4m3fn, device="cuda")
    caches = _caches()
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        gemv.m3_mx_gemv_producer(
            xq, xsf, wq, wsf, q, iq, *caches, slots, quant, EPS, *norms, rotary, positions, cfg=CFG, use_pdl=True
        )  # fmt: skip
    graph.replay()
    torch.cuda.synchronize()
    outs.append([t.contiguous().view(torch.uint8).clone() for t in (q, iq, *caches)])
    for run in outs[1:]:
        for a, b in zip(outs[0], run):
            assert torch.equal(a, b)


@pytest.mark.parametrize("tokens", [9, 16, 32])
def test_declines_past_8_rows(tokens) -> None:
    """9..32 rows (and any row count on a non-split-8 config): nothing is launched; the step keeps the plain GEMV +
    producer, so those rows stay bitwise today's."""
    case = _case(tokens, 40 + tokens)
    caches = _caches()
    ok, q, iq = _fused(case, caches)
    assert not ok
    for t in (q, iq, *caches):
        assert not bool(t.contiguous().view(torch.uint8).any()), "a declined call wrote output"


def test_declines_other_configs_and_shapes() -> None:
    case = _case(8, 41)
    ok, _, _ = _fused(case, _caches(), cfg=(4, 12))
    assert not ok, "split 4 cannot run the one-token-per-cluster-rank exchange"
    xq, xsf, wq, wsf, *rest = case
    narrow = (xq, xsf, wq[: 38 * 128].contiguous(), wsf[: 38 * 128 * K // 32].contiguous(), *rest)
    ok, _, _ = _fused(narrow, _caches())
    assert not ok, "only the 39-head packed projection"


def test_fold_rows_and_kill_switch(monkeypatch) -> None:
    """The dispatch: the fold serves the packed projection at 1..8 rows on its split-8 override, nothing past 8 rows or
    on any other projection, and nothing when killed."""
    for rows in range(1, 9):
        assert modeling._fold_cfg(N, K, rows) == CFG
    for rows in range(9, 33):
        assert modeling._fold_cfg(N, K, rows) is None
    assert modeling._fold_cfg(*modeling.QKV, 4) is None
    monkeypatch.setattr(modeling, "QKV_PRODUCER_FOLD", False)
    assert modeling._fold_cfg(N, K, 4) is None
