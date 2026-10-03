# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kimi_k3_noaux_tc_mxfp8_quant catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.kimi_k3_noaux_tc_mxfp8_quant import (
    kimi_k3_noaux_tc_mxfp8_quant,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    mxfp8_quantize,
)

assert torch.cuda.is_available(), "kimi_k3_noaux_tc_mxfp8_quant requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

NUM_EXPERTS = 896
TOPK = 16
HIDDEN = 3584
SF_COLS = HIDDEN // 32
TOKEN_COUNTS = (1, 2, 5, 8, 16, 33, 63, 64)


def _inputs(num_tokens: int, logit_scale: float = 2.5, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen, device="cuda") * logit_scale
    bias = torch.randn(NUM_EXPERTS, generator=gen, device="cuda") * 0.1
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    return logits, bias, hidden


def _check(logits, bias, hidden, routed_scaling_factor):
    """Compare one call against noaux_tc_op + mxfp8_quantize, the two catalog entries it fuses."""
    ids, weights, data, sf = kimi_k3_noaux_tc_mxfp8_quant(
        logits, bias, hidden, routed_scaling_factor
    )
    num_tokens = logits.shape[0]
    assert ids.shape == (num_tokens, TOPK) and ids.dtype == torch.int32
    assert weights.shape == (num_tokens, TOPK) and weights.dtype == torch.bfloat16
    assert data.shape == (num_tokens, HIDDEN) and data.dtype == torch.float8_e4m3fn
    assert sf.shape == (num_tokens, SF_COLS) and sf.dtype == torch.uint8
    assert all(t.is_contiguous() and t.device == logits.device for t in (ids, weights, data, sf))

    ref_weights, ref_ids = noaux_tc_op(logits, bias, 1, 1, TOPK, routed_scaling_factor)
    assert torch.equal(ids, ref_ids)
    # The kernel rounds each weight once, fp64 -> bf16; noaux_tc_op rounds fp64 -> fp32. Rounding that
    # fp32 to bf16 differs only where the fp32 value sits exactly on a bf16 midpoint (or where the two
    # kernels' fp32 sums of the selected scores differ in the last bit), so the gate is one bf16 ulp
    # (a relative 2^-7 covers one ulp anywhere in a binade) plus a bit-exactness budget.
    ref_bf16 = ref_weights.to(torch.bfloat16)
    torch.testing.assert_close(weights, ref_bf16, rtol=2.0**-7, atol=0.0)
    mismatched = int((weights != ref_bf16).sum())
    assert mismatched <= 1 + weights.numel() // 1000, (num_tokens, mismatched)

    ref_data, ref_sf = mxfp8_quantize(hidden, False, 512)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(sf, ref_sf.view(num_tokens, SF_COLS))
    return ids, weights, data, sf


def test_matches_noaux_tc_and_mxfp8_quantize() -> None:
    # Kimi K3's routed_scaling_factor (1.0) and a non-unit one, over the op's whole token range.
    for num_tokens in TOKEN_COUNTS:
        for routed_scaling_factor in (1.0, 2.5):
            logits, bias, hidden = _inputs(num_tokens, seed=num_tokens)
            _, weights, _, _ = _check(logits, bias, hidden, routed_scaling_factor)
            torch.testing.assert_close(
                weights.float().sum(-1),
                torch.full((num_tokens,), routed_scaling_factor, device="cuda"),
                rtol=16 * 2.0**-8,  # 16 bf16 addends
                atol=0.0,
            )


def test_tied_and_saturated_logits() -> None:
    # Ties (logits from 4 values, a zero bias) keep the stable descending order noaux_tc_op pins, and
    # logits beyond the sigmoid's saturation points route as they do there.
    gen = torch.Generator(device="cuda").manual_seed(7)
    levels = torch.tensor([-1.0, 0.0, 0.5, 2.0], device="cuda")
    for num_tokens in (1, 17, 64):
        logits = levels[
            torch.randint(0, 4, (num_tokens, NUM_EXPERTS), generator=gen, device="cuda")
        ]
        bias = torch.zeros(NUM_EXPERTS, device="cuda")
        hidden = torch.randn(num_tokens, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
        _check(logits, bias, hidden, 1.0)
        saturated = torch.randn(num_tokens, NUM_EXPERTS, generator=gen, device="cuda") * 40.0
        _check(saturated, torch.randn(NUM_EXPERTS, generator=gen, device="cuda"), hidden, 1.0)


def test_hidden_extremes() -> None:
    # Large and tiny row magnitudes and all-zero blocks take the same path as mxfp8_quantize.
    for scale in (1e-3, 1.0, 1e4):
        logits, bias, hidden = _inputs(8, seed=3)
        hidden = (hidden.float() * scale).to(torch.bfloat16)
        hidden[:, :64] = 0
        _check(logits, bias, hidden, 1.0)


def test_cuda_graph_replay() -> None:
    # The decode path calls this op inside a captured step: replay with new inputs equals eager.
    for num_tokens in (1, 8, 64):
        logits, bias, hidden = _inputs(num_tokens, seed=100 + num_tokens)
        kimi_k3_noaux_tc_mxfp8_quant(logits, bias, hidden, 1.0)  # warm up outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = kimi_k3_noaux_tc_mxfp8_quant(logits, bias, hidden, 1.0)
        for seed in (1, 2, 3):
            new_logits, new_bias, new_hidden = _inputs(num_tokens, seed=200 + seed)
            logits.copy_(new_logits)
            bias.copy_(new_bias)
            hidden.copy_(new_hidden)
            graph.replay()
            eager = kimi_k3_noaux_tc_mxfp8_quant(logits, bias, hidden, 1.0)
            for replayed, expected in zip(outs, eager):
                assert torch.equal(replayed.view(torch.uint8), expected.view(torch.uint8))


def test_input_not_mutated_and_deterministic() -> None:
    logits, bias, hidden = _inputs(64, seed=5)
    copies = [t.clone() for t in (logits, bias, hidden)]
    first = kimi_k3_noaux_tc_mxfp8_quant(logits, bias, hidden, 1.0)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        second = kimi_k3_noaux_tc_mxfp8_quant(logits, bias, hidden, 1.0)
    torch.cuda.current_stream().wait_stream(side)
    for a, b in zip(first, second):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    for t, c in zip((logits, bias, hidden), copies):
        assert torch.equal(t, c)


def test_rejects_out_of_contract() -> None:
    logits, bias, hidden = _inputs(8, seed=9)
    bad_calls = {
        "0 tokens": (logits[:0], bias, hidden[:0]),
        "65 tokens": _inputs(65),
        "bf16 logits": (logits.bfloat16(), bias, hidden),
        "bf16 bias": (logits, bias.bfloat16(), hidden),
        "fp32 hidden": (logits, bias, hidden.float()),
        "895 experts": (logits[:, :895].contiguous(), bias[:895].contiguous(), hidden),
        "hidden 3583": (logits, bias, hidden[:, :3583].contiguous()),
        "row-count mismatch": (logits, bias, hidden[:4].contiguous()),
        "1-D logits": (logits[0].contiguous(), bias, hidden),
        "strided logits": (torch.randn(8, 2 * NUM_EXPERTS, device="cuda")[:, ::2], bias, hidden),
        "strided hidden": (
            logits,
            bias,
            torch.randn(8, 2 * HIDDEN, device="cuda").bfloat16()[:, ::2],
        ),
        "CPU bias": (logits, bias.cpu(), hidden),
    }
    for name, args in bad_calls.items():
        try:
            kimi_k3_noaux_tc_mxfp8_quant(*args, 1.0)
        except RuntimeError:
            continue
        raise AssertionError(f"{name}: the op accepted an out-of-contract call")
