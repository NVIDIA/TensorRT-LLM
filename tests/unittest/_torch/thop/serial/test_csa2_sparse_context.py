# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``trtllm::csa2_sparse_context``: FP8 sparse context attention over the CSA2 shared bank."""

import time

import pytest
import torch

import tensorrt_llm  # noqa: F401  (loads the torch.ops.trtllm library)

HEADS = 64
DIM = 512
WINDOW = 128
TOPK = 512
WIDTH = WINDOW + TOPK
SCALE = DIM**-0.5
LOG2E = 1.4426950408889634
E4M3 = torch.float8_e4m3fn

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.ops.trtllm.csa2_sparse_context_supported(),
    reason="csa2_sparse_context requires an SM100-family GPU",
)


def _quantize(q, scale):
    """The context FP8 quantizer: FP32 product, round to nearest even, saturate to +-448, NaN kept."""
    return (q.float() * scale.float()).clamp(-448.0, 448.0).to(E4M3)


def _case(tokens, seed, *, lens=None, rows=4000, q_std=0.5, kv_std=1.0, q_scale=0.5, sinks=True):
    """Shared-bank operands: a bank whose row 0 is zero, a contiguous 128-row window plus random
    rows per query (-1 after its length), BF16 Q with a power-of-two quant scale."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    bank = (torch.randn(rows, DIM, device="cuda", generator=g) * kv_std).clamp(-448, 448).to(E4M3)
    bank[0] = 0
    if lens is None:
        lens = torch.randint(1, WIDTH + 1, (tokens,), device="cuda", generator=g)
    lens = lens.to(torch.int32)
    idx = torch.randint(1, rows, (tokens, WIDTH), device="cuda", generator=g, dtype=torch.int32)
    base = torch.randint(1, rows - WINDOW, (tokens, 1), device="cuda", generator=g)
    idx[:, :WINDOW] = (base + torch.arange(WINDOW, device="cuda")).int()
    column = torch.arange(WIDTH, device="cuda")[None, :]
    idx = torch.where(column < lens[:, None], idx, -1).contiguous()
    q = (torch.randn(tokens, HEADS, DIM, device="cuda", generator=g) * q_std).bfloat16()
    dequant = 1.0 / q_scale
    bmm1 = dequant * dequant * SCALE
    return dict(
        bank=bank,
        idx=idx,
        lens=lens,
        q=q,
        q_scale=torch.tensor([q_scale], device="cuda"),
        bmm1=torch.tensor([bmm1, bmm1 * LOG2E], device="cuda"),
        bmm2=torch.tensor([dequant], device="cuda"),
        sinks=torch.randn(HEADS, device="cuda", generator=g) if sinks else None,
    )


def _run(case, *, e4m3=False, out=None):
    q = case["q"]
    if out is None:
        out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    torch.ops.trtllm.csa2_sparse_context(
        case["bank"],
        _quantize(q, case["q_scale"]) if e4m3 else q,
        None if e4m3 else case["q_scale"],
        case["idx"],
        case["lens"],
        case["bmm1"],
        case["bmm2"],
        case["sinks"],
        out,
    )
    return out


def _wait(what, timeout=60.0):
    """Fail instead of hanging the session when a launch deadlocks."""
    event = torch.cuda.Event()
    event.record()
    start = time.time()
    while not event.query():
        if time.time() - start > timeout:
            pytest.fail(f"{what}: the fused context kernel did not finish within {timeout} s")
        time.sleep(0.002)


def _gather(case, start, end, width):
    idx = case["idx"][start:end].long()
    lens = case["lens"][start:end].long().clamp(1, idx.shape[1])
    idx = idx[:, :width]
    valid = torch.arange(width, device="cuda")[None, :] < lens[:, None]
    rows = torch.where(valid & (idx >= 0) & (idx < case["bank"].shape[0]), idx, 0)
    return case["bank"][rows].float(), valid


def _reference(case, emulate=True, chunk=32):
    """Attention over the staged E4M3 operands; out-of-range keys read the zero row.

    ``emulate`` models the FP8 context kernel's softmax: an exact running max per
    128-key block, P * 448 rounded to E4M3 for PV and FP32 row sums of the
    unrounded P. Otherwise exact FP64 softmax.
    """
    q8 = _quantize(case["q"], case["q_scale"]).float()
    tokens = q8.shape[0]
    width = int(case["lens"].clamp(1, WIDTH).max())
    bmm1, bmm2 = float(case["bmm1"][0]), float(case["bmm2"][0])
    sinks = case["sinks"]
    out = torch.empty(tokens, HEADS, DIM, device="cuda")
    for start in range(0, tokens, chunk):
        end = min(tokens, start + chunk)
        if emulate:
            width_b = (width + 127) // 128 * 128
            kv, valid = _gather(case, start, end, width_b)
            if kv.shape[1] < width_b:
                pad = width_b - kv.shape[1]
                kv = torch.nn.functional.pad(kv, (0, 0, 0, pad))
                valid = torch.nn.functional.pad(valid, (0, pad))
            logits = torch.einsum("nhd,nwd->nhw", q8[start:end], kv) * (bmm1 * LOG2E)
            logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
            m = torch.full((end - start, HEADS), -torch.inf, device="cuda")
            l_sum = torch.zeros(end - start, HEADS, device="cuda")
            o = torch.zeros(end - start, HEADS, DIM, device="cuda")
            for b in range(0, width_b, 128):
                live = valid[:, b : b + 128].any(1)[:, None]
                block = logits[..., b : b + 128]
                m_new = torch.where(live, torch.maximum(m, block.amax(-1)), m)
                alpha = torch.where(live & torch.isfinite(m), torch.exp2(m - m_new), live.float())
                alpha = torch.where(live, alpha, torch.ones_like(alpha))
                p = torch.where(live[..., None], torch.exp2(block - m_new[..., None]), 0.0)
                l_sum = l_sum * alpha + p.sum(-1)
                o = o * alpha[..., None] + torch.einsum(
                    "nhw,nwd->nhd", (p * 448.0).to(E4M3).float(), kv[:, b : b + 128]
                )
                m = m_new
            sink = 0.0 if sinks is None else torch.exp2(sinks[None, :] * LOG2E - m)
            out[start:end] = o * bmm2 / (448.0 * (l_sum + sink))[..., None]
        else:
            kv, valid = _gather(case, start, end, width)
            kv = kv.double()
            logits = torch.einsum("nhd,nwd->nhw", q8[start:end].double(), kv) * bmm1
            logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
            m = logits.amax(-1)
            if sinks is not None:
                m = torch.maximum(m, sinks[None, :].double())
            p = torch.exp(logits - m[..., None])
            den = p.sum(-1)
            if sinks is not None:
                den = den + torch.exp(sinks[None, :].double() - m)
            out[start:end] = (torch.einsum("nhw,nwd->nhd", p, kv) * bmm2 / den[..., None]).float()
    return out


def _rel(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm())


def _check_fp8_reference(actual, case):
    """Agreement with the FP8 softmax model at the BF16 output-rounding floor."""
    expected = _reference(case)
    assert torch.isfinite(actual).all()
    floor = _rel(expected.bfloat16(), expected)
    assert _rel(actual, expected) <= max(2.5e-3, 2.0 * floor), (_rel(actual, expected), floor)
    live = expected.flatten(1).norm(dim=-1) > 0
    assert not actual[~live].any(), "rows without signal must be exactly zero"
    cosine = torch.nn.functional.cosine_similarity(
        actual[live].float().flatten(1), expected[live].flatten(1), dim=-1
    )
    assert cosine.numel() == 0 or cosine.min() >= 0.9995, float(cosine.min())


def _bitwise(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


@pytest.mark.parametrize("tokens", [1, 300, 2500])
@pytest.mark.parametrize("e4m3", [False, True])
@torch.inference_mode()
def test_csa2_sparse_context_matches_fp8_reference(tokens, e4m3):
    case = _case(tokens, 11 + tokens)
    out = _run(case, e4m3=e4m3)
    _wait("reference")
    _check_fp8_reference(out, case)


@torch.inference_mode()
def test_csa2_sparse_context_matches_exact_reference_like_fp8_attention():
    """Against exact FP64 softmax the error is the FP8 P rounding, not more."""
    case = _case(1024, 5)
    out = _run(case).float()
    exact = _reference(case, emulate=False)
    emulated = _reference(case)
    assert _rel(out, exact) <= 1.02 * _rel(emulated, exact) + 1e-4


@torch.inference_mode()
def test_csa2_sparse_context_without_sinks_short_and_invalid_rows():
    tokens = 700
    lens = torch.randint(1, WIDTH + 1, (tokens,), device="cuda")
    lens[:40] = 1
    lens[40:80] = torch.arange(1, 41, device="cuda") * 3
    case = _case(tokens, 3, lens=lens, sinks=False)
    # All-invalid selections: every key reads the zero row, so the output is exactly zero.
    case["idx"][100:120] = -1
    case["idx"][120:140] = case["bank"].shape[0] + 7
    out = _run(case)
    _wait("invalid rows")
    assert torch.equal(out[100:140], torch.zeros_like(out[100:140]))
    _check_fp8_reference(out, case)
    # Columns past a query's length are never read.
    garbage = case["idx"].clone()
    column = torch.arange(WIDTH, device="cuda")[None, :]
    garbage = torch.where(column < case["lens"][:, None], garbage, 123456789)
    again = _run(dict(case, idx=garbage))
    _wait("garbage tail")
    assert _bitwise(again, out)


@pytest.mark.parametrize("tokens", [300, 2501])
@torch.inference_mode()
def test_csa2_sparse_context_fp8_output_matches_inverse_rope_quantizer(tokens):
    """The o_a projection input written by the kernel is the inverse-RoPE FP8 quantizer's output for its BF16
    output, bit for bit, including all-zero outputs and the zeroed scale rows past a token count not divisible by 4."""
    case = _case(tokens, 23 + tokens)
    case["idx"][:20] = 0
    positions = torch.randint(0, 4096, (tokens,), device="cuda", dtype=torch.int32)
    cos_sin = torch.randn(4096, 64, device="cuda")
    data, scale = torch.ops.trtllm.fused_inv_rope_fp8_quant_vllm_port(
        _run(case), positions, cos_sin, 8, 8, DIM - 64, 64, 128, False
    )
    fused_data = torch.empty_like(data)
    fused_scale = torch.full_like(scale, float("nan"))
    torch.ops.trtllm.csa2_sparse_context(
        case["bank"],
        case["q"],
        case["q_scale"],
        case["idx"],
        case["lens"],
        case["bmm1"],
        case["bmm2"],
        case["sinks"],
        fused_data,
        fused_scale,
        positions,
        cos_sin,
    )
    _wait("FP8 output")
    assert torch.equal(fused_data.view(torch.uint8), data.view(torch.uint8))
    assert torch.equal(fused_scale.view(torch.int32), scale.view(torch.int32))


def _hang_case(name, tokens):
    """Inputs where no head's maximum changes for two blocks."""
    full = torch.full((tokens,), WIDTH, device="cuda", dtype=torch.int32)
    if name == "zero_q":
        case = _case(tokens, 5, lens=full, q_std=0.0)
    elif name == "zero_q_three_blocks":
        case = _case(tokens, 5, lens=torch.full_like(full, 3 * 128), q_std=0.0)
    elif name in ("max_in_first_block", "max_in_last_block"):
        case = _case(tokens, 5, lens=full, q_std=0.05, q_scale=1.0)
        g = torch.Generator(device="cuda").manual_seed(9)
        lo, hi = (0, 128) if name == "max_in_first_block" else (512, WIDTH)
        pick = torch.randint(lo, hi, (tokens, HEADS), device="cuda", generator=g)
        rows = case["idx"].long().gather(1, pick)
        case["q"] = (case["q"].float() + 4.0 * case["bank"].float()[rows] / 22.6).bfloat16()
    elif name == "multi_request":
        lens = torch.cat(
            [
                torch.arange(1, n + 1, device="cuda").clamp_max(WIDTH)
                for n in (5, 300, 127, 1, 2000, 3)
            ]
        )[:tokens]
        case = _case(lens.numel(), 5, lens=lens, q_std=0.0)
    else:
        raise ValueError(name)
    return case


@pytest.mark.parametrize(
    "name",
    [
        "zero_q",
        "zero_q_three_blocks",
        "max_in_first_block",
        "max_in_last_block",
        "multi_request",
    ],
)
@pytest.mark.parametrize("tokens", [300, 4096])
@torch.inference_mode()
def test_csa2_sparse_context_hang_regressions(name, tokens):
    case = _hang_case(name, tokens)
    outputs = []
    for launch in range(3):
        outputs.append(_run(case))
        _wait(f"{name} launch {launch}", timeout=30.0)
    assert all(_bitwise(out, outputs[0]) for out in outputs[1:])
    _check_fp8_reference(outputs[0], case)


def _bf16_patterns(tokens, seed):
    """Every finite BF16 bit pattern (denormals, saturating magnitudes, signed zeros) plus random values."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(tokens, HEADS, DIM, device="cuda", generator=g) * 3.0).bfloat16()
    bits = (
        torch.arange(1 << 16, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    )
    bits = torch.where(torch.isnan(bits), torch.zeros_like(bits), bits)
    flat = q.view(-1)
    flat[: bits.numel()] = bits[torch.randperm(bits.numel(), device="cuda", generator=g)]
    return q


@pytest.mark.parametrize("q_scale", [1.0, 0.5, 0.125, 2.0])
@torch.inference_mode()
def test_csa2_sparse_context_quantizes_q_like_the_context_quantizer(q_scale):
    """BF16 Q quantized in the kernel gives the same output bits as Q quantized beforehand."""
    case = _case(512, 21, q_scale=q_scale, kv_std=0.25)
    case["q"] = _bf16_patterns(512, 22)
    # Infinite and huge Q saturate to +-448 in both paths; this BMM1 scale keeps the logits finite
    # but spread over several units, so the output depends on every Q byte.
    case["bmm1"] = torch.tensor([2e-3, 2e-3 * LOG2E], device="cuda")
    fused = _run(case)
    staged = _run(case, e4m3=True)
    _wait("quantization")
    assert torch.isfinite(fused.float()).all()
    assert _bitwise(fused, staged)
    # A large logit spread makes every output sensitive to single Q bytes.
    case = _case(2048, 23, q_std=2.0, q_scale=q_scale)
    assert _bitwise(_run(case), _run(case, e4m3=True))


@torch.inference_mode()
def test_csa2_sparse_context_back_to_back_launches_and_graph_replay():
    cases = [_case(3000, 31), _case(157, 32, sinks=False), _case(1, 33)]
    expected = [_run(case) for case in cases]
    _wait("eager")
    outputs = [torch.empty_like(out) for out in expected]
    for _ in range(10):
        for case, out in zip(cases, outputs):
            _run(case, out=out)
    _wait("back to back")
    assert all(_bitwise(out, exp) for out, exp in zip(outputs, expected))
    case = cases[0]
    out = torch.empty_like(expected[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(case, out=out)
    graph.replay()
    _wait("graph")
    assert _bitwise(out, expected[0])
    case["q"].mul_(-1.0)
    graph.replay()
    _wait("graph after update")
    assert _bitwise(out, _run(case))
