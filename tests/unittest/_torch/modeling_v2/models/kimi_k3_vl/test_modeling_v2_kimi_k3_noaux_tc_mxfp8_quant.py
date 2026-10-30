# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the kimi_k3_noaux_tc_mxfp8_quant catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe import noaux_tc_op as noaux_tc
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization import (
    mxfp8_quantize as mxfp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.moe.kimi_k3_noaux_tc_mxfp8_quant import (
    kimi_k3_noaux_tc_mxfp8_quant,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "kimi_k3_noaux_tc_mxfp8_quant requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

NUM_EXPERTS = 896
TOPK = 16
HIDDEN = 3584
SF_COLS = HIDDEN // 32


def _inputs(num_tokens: int, logit_scale: float = 2.5, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen, device="cuda") * logit_scale
    bias = torch.randn(NUM_EXPERTS, generator=gen, device="cuda") * 0.1
    hidden = torch.randn(num_tokens, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    return logits, bias, hidden


def _check(logits, bias, hidden, routed_scaling_factor):
    """Compare one call against noaux_tc_op + mxfp8_quantize, the two catalog entries it fuses."""
    with validating(kimi_k3_noaux_tc_mxfp8_quant):
        ids, weights, data, sf = kimi_k3_noaux_tc_mxfp8_quant(
            logits, bias, hidden, routed_scaling_factor
        )
    num_tokens = logits.shape[0]
    assert ids.shape == (num_tokens, TOPK) and ids.dtype == torch.int32
    assert weights.shape == (num_tokens, TOPK) and weights.dtype == torch.bfloat16
    assert data.shape == (num_tokens, HIDDEN) and data.dtype == torch.float8_e4m3fn
    assert sf.shape == (num_tokens, SF_COLS) and sf.dtype == torch.uint8
    assert all(t.is_contiguous() and t.device == logits.device for t in (ids, weights, data, sf))
    ref_ids, ref_weights, ref_data, ref_sf = kimi_k3_noaux_tc_mxfp8_quant.reference(
        logits, bias, hidden, routed_scaling_factor
    )
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(weights, ref_weights, rtol=2.0**-7, atol=0.0)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(sf, ref_sf)

    ref_weights, ref_ids = noaux_tc.noaux_tc_op(logits, bias, 1, 1, TOPK, routed_scaling_factor)
    assert torch.equal(ids, ref_ids)
    # The kernel rounds each weight once, fp64 -> bf16; noaux_tc_op rounds fp64 -> fp32. Rounding that
    # fp32 to bf16 differs only where the fp32 value sits exactly on a bf16 midpoint (or where the two
    # kernels' fp32 sums of the selected scores differ in the last bit), so the gate is one bf16 ulp
    # (a relative 2^-7 covers one ulp anywhere in a binade) plus a bit-exactness budget.
    ref_bf16 = ref_weights.to(torch.bfloat16)
    torch.testing.assert_close(weights, ref_bf16, rtol=2.0**-7, atol=0.0)
    mismatched = int((weights != ref_bf16).sum())
    assert mismatched <= 1 + weights.numel() // 1000, (num_tokens, mismatched)

    ref_data, ref_sf = mxfp8.mxfp8_quantize(hidden, False, 512)
    assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
    assert torch.equal(sf, ref_sf.view(num_tokens, SF_COLS))
    return ids, weights, data, sf


@pytest.mark.parametrize(
    "cell",
    kimi_k3_noaux_tc_mxfp8_quant.CELLS,
    ids=[c.why[:40] for c in kimi_k3_noaux_tc_mxfp8_quant.CELLS],
)
def test_certified_cells(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    num_tokens, routed_scaling_factor = spec["tokens"], spec["routed_scaling_factor"]
    logits, bias, hidden = _inputs(num_tokens, seed=seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    if spec["logits"] == "tied":
        levels = torch.tensor([-1.0, 0.0, 0.5, 2.0], device="cuda")
        logits = levels[
            torch.randint(0, 4, (num_tokens, NUM_EXPERTS), generator=gen, device="cuda")
        ]
        bias = torch.zeros(NUM_EXPERTS, device="cuda")
    elif spec["logits"] == "saturated":
        logits = torch.randn(num_tokens, NUM_EXPERTS, generator=gen, device="cuda") * 40.0
        bias = torch.randn(NUM_EXPERTS, generator=gen, device="cuda")
    if "hidden_scale" in spec:
        hidden = (hidden.float() * spec["hidden_scale"]).to(torch.bfloat16)
    if spec.get("zero_blocks"):
        hidden[:, :64] = 0
    _, weights, _, _ = _check(logits, bias, hidden, routed_scaling_factor)
    torch.testing.assert_close(
        weights.float().sum(-1),
        torch.full((num_tokens,), routed_scaling_factor, device="cuda"),
        rtol=16 * 2.0**-8,  # 16 bf16 addends
        atol=0.0,
    )


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
        # Free this capture's outputs before the next capture starts: rebound inside it, they would be freed inside
        # another graph's capture, which cudaMallocAsync turns into a captured free of this graph's memory.
        del outs, graph


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
