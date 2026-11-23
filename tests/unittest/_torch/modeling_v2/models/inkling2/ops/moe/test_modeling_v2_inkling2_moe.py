# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the Inkling 2 MoE entries: router, NVFP4 experts, MXFP8 experts."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.moe.inkling2_mxfp8_experts import (
    inkling2_mxfp8_experts,
    mxfp8_dequantize,
    mxfp8_quantize,
)
from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.moe.inkling2_nvfp4_experts import (
    fc1_row_order,
    fc2_input_scale,
    inkling2_nvfp4_experts,
    swizzle_block_scales,
    unswizzle_block_scales,
)
from tensorrt_llm._torch.modeling_v2.models.inkling2.ops.moe.inkling2_router import inkling2_router

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "the Inkling 2 MoE entries require a CUDA device"

GATE_SOFTCAP, UP_SOFTCAP = 4.0, 25.0


def _seed(text: str) -> int:
    return zlib.crc32(text.encode()) % 2**31


def _routing(g, tokens, experts, top_k, local, offset):
    """Distinct experts per token, about half of them on this rank; positive weights summing to 8."""
    pool = torch.cat(
        [
            torch.arange(offset, offset + local),
            torch.arange(0, offset),
            torch.arange(offset + local, experts),
        ]
    )
    pick = torch.stack([torch.randperm(2 * local, generator=g)[:top_k] for _ in range(tokens)])
    ids = pool[pick.clamp(max=experts - 1)]
    weights = torch.rand(tokens, top_k, generator=g) + 0.1
    weights = weights / weights.sum(-1, keepdim=True) * 8.0
    return ids.to(torch.int32).cuda(), weights.float().cuda()


def _rel_rms(out, ref):
    return float(((out.float() - ref.float()).pow(2).mean() / ref.float().pow(2).mean()).sqrt())


def _e4m3_scales(g, shape, lo=-4, hi=3):
    exps = torch.randint(lo, hi, shape, generator=g, device="cuda")
    return torch.exp2(exps.float()).to(torch.float8_e4m3fn).view(torch.uint8)


def _nvfp4_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    e, h, i = spec["local"], spec["hidden"], spec["inter"]
    dev = "cuda"
    w13 = torch.randint(0, 256, (e, 2 * i, h // 2), generator=g, dtype=torch.uint8, device=dev)
    w13_sf = _e4m3_scales(g, (e, 2 * i, h // 16))
    w2 = torch.randint(0, 256, (e, h, i // 2), generator=g, dtype=torch.uint8, device=dev)
    w2_sf = _e4m3_scales(g, (e, h, i // 16))
    # Global scales that keep the experts' outputs O(1) for unit-RMS inputs.
    w13_alpha = (0.02 + 0.01 * torch.rand(e, generator=g, device=dev)).float()
    w2_alpha = (0.02 + 0.01 * torch.rand(e, generator=g, device=dev)).float()
    offset = spec["rank"] * e
    x = torch.randn(spec["tokens"], h, generator=g, device=dev).to(torch.bfloat16)
    ids, weights = _routing(
        torch.Generator().manual_seed(seed),
        spec["tokens"],
        spec["experts"],
        spec["top_k"],
        e,
        offset,
    )
    return [
        x,
        ids,
        weights,
        w13,
        swizzle_block_scales(w13_sf),
        w13_alpha,
        w2,
        swizzle_block_scales(w2_sf),
        w2_alpha,
        fc2_input_scale(torch.device("cuda")),
        spec["experts"],
        offset,
        GATE_SOFTCAP,
        UP_SOFTCAP,
    ]


def _mxfp8_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    e, h, i = spec["local"], spec["hidden"], spec["inter"]
    dev = "cuda"

    def e4m3(shape):
        return (
            (64 * torch.randn(shape, generator=g, device=dev))
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )

    def exponents(shape):
        return torch.randint(127 - 14, 127 - 10, shape, generator=g, dtype=torch.uint8, device=dev)

    w13, w13_s = e4m3((e, 2 * i, h)), exponents((e, 2 * i, h // 32))
    w2, w2_s = e4m3((e, h, i)), exponents((e, h, i // 32))
    offset = spec["rank"] * e
    x = torch.randn(spec["tokens"], h, generator=g, device=dev).to(torch.bfloat16)
    ids, weights = _routing(
        torch.Generator().manual_seed(seed),
        spec["tokens"],
        spec["experts"],
        spec["top_k"],
        e,
        offset,
    )
    return [x, ids, weights, w13, w13_s, w2, w2_s, offset, GATE_SOFTCAP, UP_SOFTCAP]


# --------------------------------------------------------------------------------------- router


@pytest.mark.parametrize("cell", inkling2_router.CELLS, ids=[c.why for c in inkling2_router.CELLS])
def test_router_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(inkling2_router)
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    x = torch.randn(spec["tokens"], spec["hidden"], generator=g, device="cuda").to(torch.bfloat16)
    weight = (0.015 * torch.randn(spec["experts"], spec["hidden"], generator=g, device="cuda")).to(
        torch.bfloat16
    )
    bias = 0.05 * torch.randn(spec["experts"], generator=g, device="cuda")
    ids, weights = inkling2_router(x, weight, bias, spec["top_k"], spec["route_scale"])
    ref_ids, ref_weights = inkling2_router.reference(
        x, weight, bias, spec["top_k"], spec["route_scale"]
    )
    # The selection is a set; order the two by expert id before comparing.
    order, ref_order = ids.sort(-1), ref_ids.sort(-1)
    assert torch.equal(order.values, ref_order.values)
    torch.testing.assert_close(
        weights.gather(1, order.indices.long()),
        ref_weights.gather(1, ref_order.indices.long()),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        weights.sum(-1), torch.full_like(weights[:, 0], 8.0), rtol=1e-5, atol=1e-5
    )


# ----------------------------------------------------------------------------------------- NVFP4


def test_fc1_row_order_and_swizzle_round_trip() -> None:
    order = fc1_row_order(2304, torch.device("cpu"))
    # The checkpoint interleaves gate (even) and up (odd) rows. Kernel group g (128 rows) holds the
    # up rows of gate/up pairs 64 g .. 64 g + 63, then their gate rows: every group, in order.
    expected = []
    for group in range(2304 // 64):
        pairs = range(64 * group, 64 * group + 64)
        expected += [2 * p + 1 for p in pairs] + [2 * p for p in pairs]
    assert order.tolist() == expected
    scales = torch.randint(0, 256, (3, 256, 288), dtype=torch.uint8)
    assert torch.equal(unswizzle_block_scales(swizzle_block_scales(scales)), scales)


def test_swizzle_matches_flashinfer_block_scale_interleave() -> None:
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    scales = torch.randint(0, 256, (2, 256, 144), dtype=torch.uint8, device="cuda")
    assert torch.equal(
        swizzle_block_scales(scales).reshape(-1), block_scale_interleave(scales).reshape(-1)
    )


@pytest.mark.parametrize(
    "cell", inkling2_nvfp4_experts.CELLS, ids=[c.why for c in inkling2_nvfp4_experts.CELLS]
)
def test_nvfp4_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(inkling2_nvfp4_experts)
    args = _nvfp4_args(cell.spec, _seed(cell.why))
    out = inkling2_nvfp4_experts(*args)
    ref = inkling2_nvfp4_experts.reference(*args)
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    # The kernel requantizes the fp32 SiTU output, the reference its bf16 rounding: a few block
    # scale choices and E2M1 roundings differ, each by a quantization step.
    assert _rel_rms(out, ref) < 0.02, _rel_rms(out, ref)


def test_nvfp4_runs_under_inference_mode() -> None:
    """The engine runs every forward under inference mode, a layer's first call included."""
    args = _nvfp4_args(dict(inkling2_nvfp4_experts.CELLS[1].spec), 23)
    ref = inkling2_nvfp4_experts.reference(*args)
    with torch.inference_mode():
        out = inkling2_nvfp4_experts(*args)
    assert _rel_rms(out, ref) < 0.02, _rel_rms(out, ref)


def test_nvfp4_swapped_gate_and_up_is_visible() -> None:
    spec = dict(inkling2_nvfp4_experts.CELLS[1].spec)
    args = _nvfp4_args(spec, 17)
    ref = inkling2_nvfp4_experts.reference(*args)
    swapped = list(args)
    i = args[3].shape[1] // 2
    groups = args[3].view(-1, i // 64, 2, 64, args[3].shape[-1]).flip(2).reshape(args[3].shape)
    swapped[3] = groups.contiguous()
    sf = unswizzle_block_scales(args[4])
    swapped[4] = swizzle_block_scales(
        sf.view(-1, i // 64, 2, 64, sf.shape[-1]).flip(2).reshape(sf.shape).contiguous()
    )
    assert _rel_rms(inkling2_nvfp4_experts(*swapped), ref) > 0.3


# ----------------------------------------------------------------------------------------- MXFP8


def test_align_routes_puts_each_local_route_in_one_row_of_its_expert() -> None:
    from tensorrt_llm._torch.modeling_v2.models.inkling2.kernels.mxfp8_moe import (
        BLOCK_M,
        align_routes,
    )

    ids, _ = _routing(torch.Generator().manual_seed(4), 100, 384, 8, 96, 96)
    route_of_row, expert_of_block, max_blocks = align_routes(ids, 96, 96)
    rows = route_of_row.view(max_blocks, BLOCK_M).cpu()
    experts = expert_of_block.cpu()
    flat = ids.reshape(-1).cpu().long() - 96
    local = set(torch.nonzero((flat >= 0) & (flat < 96)).flatten().tolist())
    placed = []
    for b in range(max_blocks):
        routes = [r for r in rows[b].tolist() if r >= 0]
        if experts[b] < 0:
            assert not routes
        assert all(flat[r] == experts[b] for r in routes)
        placed += routes
    assert sorted(placed) == sorted(local)


def test_mxfp8_expert_with_more_routes_than_a_row_block() -> None:
    """33 routes to one expert fill a 32-row block and spill into a second: both count."""
    from tensorrt_llm._torch.modeling_v2.models.inkling2.kernels.mxfp8_moe import (
        BLOCK_M,
        align_routes,
    )

    tokens = BLOCK_M + 1
    spec = dict(local=4, hidden=256, inter=256, rank=1, tokens=tokens, experts=16, top_k=2)
    args = _mxfp8_args(spec, 31)
    offset = args[7]
    # Every token's first choice is local expert 2; its second is expert 0, on another rank.
    args[1] = torch.tensor([[offset + 2, 0]] * tokens, dtype=torch.int32, device="cuda")
    args[2] = torch.full((tokens, 2), 4.0, device="cuda")
    route_of_row, expert_of_block, max_blocks = align_routes(args[1], offset, spec["local"])
    rows = route_of_row.view(max_blocks, BLOCK_M).cpu()
    blocks = [b for b in range(max_blocks) if int(expert_of_block[b]) == 2]
    assert len(blocks) == 2
    placed = sorted(r for b in blocks for r in rows[b].tolist() if r >= 0)
    assert placed == [2 * t for t in range(tokens)], "each route exactly once"
    out = inkling2_mxfp8_experts(*args)
    ref = inkling2_mxfp8_experts.reference(*args)
    assert _rel_rms(out, ref) < 2e-3, _rel_rms(out, ref)


def test_mxfp8_refuses_a_partial_k_tile() -> None:
    """Neither GEMM masks a partial 128-wide K tile, so such a size is refused before launch."""
    spec = dict(local=2, hidden=256, inter=160, rank=0, tokens=4, experts=8, top_k=2)
    args = _mxfp8_args(spec, 7)
    with pytest.raises(ValueError, match="multiples of 128"):
        inkling2_mxfp8_experts(*args)
    with pytest.raises(AssertionError, match="128-wide K tile"):
        with validating(inkling2_mxfp8_experts):
            inkling2_mxfp8_experts(*args)


def test_mxfp8_quantize_round_trip() -> None:
    x = torch.randn(64, 256, device="cuda") * torch.logspace(-3, 3, 256, device="cuda")
    values, exps = mxfp8_quantize(x)
    back = mxfp8_dequantize(values, exps)
    # E4M3 keeps 3 mantissa bits: half an ulp is 1/16 of the block's power-of-two range.
    blocks = x.view(64, 8, 32).abs().amax(-1, keepdim=True)
    assert ((back - x).view(64, 8, 32).abs() <= blocks / 16 + 1e-30).all()
    assert ((back.view(64, 8, 32).abs().amax(-1, keepdim=True) / blocks) > 0.5).all()


@pytest.mark.parametrize(
    "cell", inkling2_mxfp8_experts.CELLS, ids=[c.why for c in inkling2_mxfp8_experts.CELLS]
)
def test_mxfp8_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(inkling2_mxfp8_experts)
    args = _mxfp8_args(cell.spec, _seed(cell.why))
    out = inkling2_mxfp8_experts(*args)
    ref = inkling2_mxfp8_experts.reference(*args)
    # Same products; the bf16 GEMMs accumulate in another order, which moves a few SiTU outputs
    # across an E4M3 rounding boundary before the down projection.
    assert _rel_rms(out, ref) < 2e-3, _rel_rms(out, ref)


@pytest.mark.parametrize("case", ["bf16_bias", "fp16_x"])
def test_router_guard(case) -> None:
    x = torch.randn(4, 64, device="cuda").to(torch.bfloat16)
    weight = torch.randn(16, 64, device="cuda").to(torch.bfloat16)
    bias = torch.zeros(16, device="cuda")
    if case == "bf16_bias":
        bias = bias.to(torch.bfloat16)
    else:
        x = x.to(torch.float16)
    with pytest.raises(AssertionError):
        with validating(inkling2_router):
            inkling2_router(x, weight, bias, 4, 8.0)
