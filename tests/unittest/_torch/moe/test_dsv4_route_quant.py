# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 routing fused with NVFP4 input quantization against the unfused launches."""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV4MoeRoutingMethod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (10, 0)
    or not IS_CUTLASS_DSL_AVAILABLE,
    reason="Requires Blackwell or newer and the CuTe DSL",
)

# DeepSeek-V4.1-Flash: 384 experts, top-6, hidden 5120, routed scaling 2.5.
NUM_EXPERTS, TOP_K, HIDDEN, SCALING = 384, 6, 5120, 2.5


def _routing(bias):
    return DeepSeekV4MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=SCALING,
        callable_e_score_correction_bias=lambda: bias,
        callable_tid2eid=lambda: None,
        is_hashed=False,
    )


def _inputs(num_tokens, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(num_tokens, NUM_EXPERTS, device="cuda", generator=generator) * 4
    bias = torch.randn(NUM_EXPERTS, device="cuda", generator=generator) * 0.5
    x = (torch.randn(num_tokens, HIDDEN, device="cuda", generator=generator) * 3).bfloat16()
    global_scale = torch.tensor([448.0 * 6.0 / 20.0], device="cuda")
    return logits, bias, x, global_scale


def _unfused(routing, logits, x, global_scale, weights_dtype):
    ids, weights = routing.apply(logits)
    x_fp4, x_sf = torch.ops.trtllm.fp4_quantize(x, global_scale, 16, False, False)
    return ids, weights.to(weights_dtype), x_fp4.view(torch.uint8), x_sf.view(x.shape[0], -1)


def _fused(routing, logits, x, global_scale, weights_dtype):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dsv4_route_quant import (
        dsv4_route_nvfp4_quant,
    )

    ids, weights, x_fp4, x_sf, tables = dsv4_route_nvfp4_quant(
        logits,
        routing.e_score_correction_bias,
        x,
        global_scale,
        routing.top_k,
        routing.routed_scaling_factor,
        weights_dtype,
    )
    assert tables is None
    return ids, weights, x_fp4, x_sf


def _assert_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert got.dtype == want.dtype and got.shape == want.shape
        torch.testing.assert_close(got, want, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("num_tokens", [1, 6, 12, 192, 1000])
@pytest.mark.parametrize("weights_dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_fused_route_quant_is_bitwise_to_unfused(num_tokens, weights_dtype):
    logits, bias, x, global_scale = _inputs(num_tokens)
    routing = _routing(bias)
    expected = _unfused(routing, logits, x, global_scale, weights_dtype)
    _assert_equal(_fused(routing, logits, x, global_scale, weights_dtype), expected)


@torch.inference_mode()
def test_fused_route_quant_ties_nans_and_zero_groups():
    logits, bias, x, global_scale = _inputs(8, seed=1)
    # Ties, NaN and infinite logits, an all-zero group and subnormal inputs.
    logits[0] = 1.0
    logits[1, 5] = float("nan")
    logits[2, :7] = float("inf")
    bias[100:110] = 0.25
    logits[3, 100:110] = 3.0
    x[4, :32] = 0
    x[5, 16:32] = 1e-40
    routing = _routing(bias)
    expected = _unfused(routing, logits, x, global_scale, torch.float32)
    _assert_equal(_fused(routing, logits, x, global_scale, torch.float32), expected)


@torch.inference_mode()
def test_fused_route_quant_replays_in_a_cuda_graph():
    logits, bias, x, global_scale = _inputs(12, seed=2)
    routing = _routing(bias)
    _fused(routing, logits, x, global_scale, torch.bfloat16)  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _fused(routing, logits, x, global_scale, torch.bfloat16)
    logits.copy_(torch.randn_like(logits) * 4)
    x.copy_(torch.randn_like(x) * 3)
    graph.replay()
    expected = _unfused(routing, logits, x, global_scale, torch.bfloat16)
    _assert_equal(outputs, expected)


def _expected_tables(ids: torch.Tensor, tile: int):
    """Each expert's padded rows, the CTA experts and the CTA row limits, in expert order."""
    counts = torch.bincount(ids.flatten().long(), minlength=NUM_EXPERTS).tolist()
    rows, cta_expert, cta_limit, first = {}, [], [], 0
    for expert, count in enumerate(counts):
        if count == 0:
            continue
        ctas = (count + tile - 1) // tile
        rows[expert] = set(range(first * tile, first * tile + count))
        for c in range(ctas):
            cta_expert.append(expert)
            cta_limit.append(min((first + c + 1) * tile, first * tile + count))
        first += ctas
    return rows, cta_expert, cta_limit


def _assert_trtllm_gen_layout(ids: torch.Tensor, tables: torch.Tensor):
    """``tables`` are TRTLLM-Gen's routing tables of ``ids``, up to the row order in an expert."""
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dsv4_route_quant import ROUTING_TABLE_TILES

    e2p, p2t, cta_expert, cta_limit, counts = tables.cpu().unbind(1)
    flat = ids.flatten().cpu().tolist()
    for t, tile in enumerate(ROUTING_TABLE_TILES):
        rows, experts, limits = _expected_tables(ids.cpu(), tile)
        num_ctas = len(experts)
        assert counts[t, :2].tolist() == [num_ctas, num_ctas * tile]
        assert cta_expert[t, :num_ctas].tolist() == experts
        assert cta_limit[t, :num_ctas].tolist() == limits
        permuted = e2p[t, : len(flat)].tolist()
        for expert, block in rows.items():
            assert {p for p, e in zip(permuted, flat) if e == expert} == block
        assert [p2t[t, p].item() for p in permuted] == [i // TOP_K for i in range(len(flat))]


def _clustered_inputs(num_tokens, seed=3):
    """Inputs routed to few experts, so experts share tokens and some need several CTAs."""
    logits, bias, x, global_scale = _inputs(num_tokens, seed)
    logits[:, :8] += 12.0
    return logits, bias, x, global_scale


def _route_with_tables(logits, bias, x, global_scale, weights_dtype=torch.bfloat16):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dsv4_route_quant import (
        dsv4_route_nvfp4_quant,
    )

    return dsv4_route_nvfp4_quant(
        logits, bias, x, global_scale, TOP_K, SCALING, weights_dtype, with_routing_tables=True
    )


@pytest.mark.parametrize("num_tokens", [1, 6, 8])
@torch.inference_mode()
def test_routing_tables_match_trtllm_gen_layout(num_tokens):
    logits, bias, x, global_scale = _clustered_inputs(num_tokens)
    *outputs, tables = _route_with_tables(logits, bias, x, global_scale)
    _assert_trtllm_gen_layout(outputs[0], tables)
    # The routing and the quantized input are those of the launch without tables.
    expected = _fused(_routing(bias), logits, x, global_scale, torch.bfloat16)
    _assert_equal(outputs, expected)


@torch.inference_mode()
def test_routing_tables_replay_in_a_cuda_graph():
    logits, bias, x, global_scale = _clustered_inputs(8, seed=4)
    _route_with_tables(logits, bias, x, global_scale)  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        *outputs, tables = _route_with_tables(logits, bias, x, global_scale)
    logits.copy_(torch.randn_like(logits) * 4)
    graph.replay()
    _assert_equal(outputs, _fused(_routing(bias), logits, x, global_scale, torch.bfloat16))
    _assert_trtllm_gen_layout(outputs[0], tables)


@pytest.mark.parametrize("do_finalize", [True, False])
@pytest.mark.parametrize("num_tokens", [1, 6, 8])
@torch.inference_mode()
def test_moe_with_routing_tables_is_bitwise_to_its_own_routing(num_tokens, do_finalize):
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dsv4_route_quant import ROUTING_TABLE_TILES

    torch.manual_seed(7 + num_tokens)
    inter = 576
    ids, weights, x_fp4, x_sf, tables = _route_with_tables(*_inputs(num_tokens, seed=num_tokens))

    def packed(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    def scales(*shape):  # E4M3 in [0.25, 2]
        codes = torch.randint(0x28, 0x41, shape, dtype=torch.uint8, device="cuda")
        return codes.view(torch.float8_e4m3fn)

    w13, w13_sf = (
        packed(NUM_EXPERTS, 2 * inter, HIDDEN // 2),
        scales(NUM_EXPERTS, 2 * inter, HIDDEN // 16),
    )
    w2, w2_sf = packed(NUM_EXPERTS, HIDDEN, inter // 2), scales(NUM_EXPERTS, HIDDEN, inter // 16)
    one = torch.full((NUM_EXPERTS,), 1e-4, device="cuda")
    clamp = torch.full((NUM_EXPERTS,), 1e5, device="cuda")
    runner = torch.classes.trtllm.FP4BlockScaleMoERunner(0)

    def run(tactic, routing_tables):
        return runner.run_moe(
            None, None, x_fp4, x_sf.flatten().view(torch.float8_e4m3fn), w13, w13_sf,
            None, None, None, clamp, w2, w2_sf, None, one, one, one, NUM_EXPERTS, TOP_K,
            None, None, inter, 0, NUM_EXPERTS, None, 1, do_finalize, tactic, weights, ids,
            None, routing_tables, list(ROUTING_TABLE_TILES),
        )  # fmt: skip

    tactics = runner.get_valid_configs(TOP_K, HIDDEN, inter, NUM_EXPERTS, num_tokens)
    for tile in sorted({tactic[0] for tactic in tactics}):
        tactic = next(t for t in tactics if t[0] == tile)
        expected, got = run(tactic, None), run(tactic, tables)
        if do_finalize:
            torch.testing.assert_close(got[0], expected[0], atol=0, rtol=0)
        else:
            if tile in ROUTING_TABLE_TILES:
                # The permutation of a run on the tables is the tables' own, named by tile.
                assert got[2].shape == (0, tile)
                got = [got[0], got[1], tables[ROUTING_TABLE_TILES.index(tile), 0]]
            # Rows of one expert may sit elsewhere in its block: compare them by expanded index.
            e2p = [out[2].flatten()[: num_tokens * TOP_K].long() for out in (expected, got)]
            rows = [out[0][e] for out, e in zip((expected, got), e2p)]
            torch.testing.assert_close(rows[1], rows[0], atol=0, rtol=0)
