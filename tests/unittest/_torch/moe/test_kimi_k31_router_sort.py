# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The CuTe routing tail writing the TRTLLM-Gen MXFP4 runner's routing (``KIMI_K31_ROUTER_SORT``).

* ``k3_route_tail.run_k3_route_runner``: top-16 ids equal ``noaux_tc_op``'s; the BF16 weights equal
  ``noaux_tc_op``'s FP32 weights cast to BF16, or (``bf16_weights_f64``) the weights of
  ``kimi_k3_noaux_tc_mxfp8_quant``; the routing is the runner's layout (per local expert
  ``ceil(count / tile)`` tiles in local-expert order, rows of an expert in token order, the CTA
  arrays, padded size and CTA count, ``-1`` for other ranks' experts, the token of each row), for
  the cluster kernel (<= 256 tokens) and the multi-CTA kernel above.
* Runner: ``run_moe_unfinalized_prerouted`` / ``run_moe_prerouted`` over that routing give the same
  FC2 rows / finalized output, bit for bit, as the runner routing the same top-k itself, for every
  tactic tile; the prerouted ops fall back to the runner's routing on another tile.
* The runner ops record the tile of their tuned tactic (``mxfp4_prerouted_tile``).
* KIMI_K31_ROUTER_TAIL_DEP serves backends behind a token dispatch only when they read BF16 routing scales.
"""

from types import SimpleNamespace

import pytest
import torch

NUM_EXPERTS = 896
TOP_K = 16
LATENT = 3584


def _supported() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) not in {
        (10, 0),
        (10, 3),
    }:
        return False
    try:
        import cutlass  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _supported(), reason="needs SM100/SM103 and the CuTe DSL")


def _tail():
    import tensorrt_llm  # noqa: F401  (registers the reference ops)
    from tensorrt_llm._torch.moe.kernels.blackwell import k3_route_tail

    return k3_route_tail


def _logits(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(m, NUM_EXPERTS, generator=g, device="cuda")
    bias = (torch.randn(NUM_EXPERTS, generator=g, device="cuda") * 0.05).float()
    return logits, bias


def _reference_layout(ids: torch.Tensor, tile: int, offset: int, local: int):
    flat = ids.view(-1).tolist()
    m = ids.shape[0]
    rows_of = [[] for _ in range(local)]
    for t in range(m):
        for k in range(TOP_K):
            e = flat[t * TOP_K + k] - offset
            if 0 <= e < local:
                rows_of[e].append(t * TOP_K + k)
    e2p = [-1] * (m * TOP_K)
    p2t = {}
    cta_expert, cta_limit = [], []
    tiles = 0
    for e in range(local):
        start = tiles * tile
        for r, expanded in enumerate(rows_of[e]):
            e2p[expanded] = start + r
            p2t[start + r] = expanded // TOP_K
        n = (len(rows_of[e]) + tile - 1) // tile
        for c in range(n):
            cta_expert.append(e)
            cta_limit.append(min(start + (c + 1) * tile, start + len(rows_of[e])))
        tiles += n
    return e2p, p2t, cta_expert, cta_limit, tiles


def _route(m: int, seed: int, tile: int, offset: int, local: int, f64: bool, wide_from: int = 257):
    kt = _tail()
    logits, bias = _logits(m, seed)
    if m > kt.MAX_M or m >= wide_from:
        kt.wide_workspace(logits.device)
    ids, vals, routed, _ = kt.run_k3_route_runner(
        logits, m, bias, 1.0, offset, local, NUM_EXPERTS, tile, f64, wide_from=wide_from
    )
    ref_vals, ref_ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
    assert torch.equal(ids, ref_ids.to(torch.int32))
    assert vals.dtype == torch.bfloat16
    if not f64:
        assert torch.equal(
            vals.view(torch.int16), ref_vals.float().to(torch.bfloat16).view(torch.int16)
        )
    return logits, bias, ids, vals, routed


# (local_expert_offset, local_num_experts): TEP8 EP8, TP4xEP2, no EP.
_LAYOUTS = [(3 * 112, 112), (448, 448), (0, 896)]


@pytest.mark.parametrize("m", [1, 65, 256])
@pytest.mark.parametrize("tile", [8, 128])
@pytest.mark.parametrize("offset,local", _LAYOUTS)
def test_runner_layout(m, tile, offset, local, wide_from=257):
    _, _, ids, _, routed = _route(m, m * 31 + tile + offset, tile, offset, local, False, wide_from)
    e2p, p2t, total, cta_expert, cta_limit, num_ctas = routed
    ref_e2p, ref_p2t, ref_expert, ref_limit, ref_tiles = _reference_layout(ids, tile, offset, local)
    assert e2p.tolist()[: m * TOP_K] == ref_e2p
    assert int(num_ctas.item()) == ref_tiles
    assert int(total.item()) == ref_tiles * tile
    assert cta_expert.tolist()[:ref_tiles] == ref_expert
    assert cta_limit.tolist()[:ref_tiles] == ref_limit
    p2t_list = p2t.tolist()
    assert all(p2t_list[row] == token for row, token in ref_p2t.items())


@pytest.mark.parametrize("m", [257, 288, 448, 896])
@pytest.mark.parametrize("tile", [8, 32, 128])
@pytest.mark.parametrize("offset,local", [(0, 112), (5 * 112, 112), (0, 56), (7 * 56, 56)])
def test_runner_layout_wide(m, tile, offset, local):
    kt = _tail()
    if not kt.runner_sort_fits(m, local):
        pytest.skip("outside the multi-CTA sort's tables")
    test_runner_layout(m, tile, offset, local)


@pytest.mark.parametrize("m", [65, 112, 144, 224, 256])
@pytest.mark.parametrize("tile", [8, 32, 64])
@pytest.mark.parametrize("offset,local", [(0, 112), (3 * 112, 112), (7 * 56, 56)])
def test_runner_layout_wide_below_256(m, tile, offset, local):
    """The multi-CTA kernel (``wide_from``) also serves cluster-sized row counts with the same layout."""
    kt = _tail()
    if not kt.wide_sort_fits(m, local):
        pytest.skip("outside the multi-CTA sort's tables")
    test_runner_layout(m, tile, offset, local, wide_from=1)


@pytest.mark.parametrize("m", [1, 7, 28, 42, 64])
def test_small_m_matches_fused_route_quant(m):
    """At <= 64 rows the tail's FP64-rounded BF16 weights and the separate MXFP8 quantization equal
    ``kimi_k3_noaux_tc_mxfp8_quant`` (which the routing tail replaces there)."""
    logits, bias, ids, vals, _ = _route(m, 77 + m, 16, 0, 112, True)
    torch.manual_seed(m)
    x = (torch.randn(m, LATENT, device="cuda") * 3).to(torch.bfloat16)
    ref_ids, ref_vals, ref_q, ref_sf = torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
        logits, bias, x, 1.0
    )
    assert torch.equal(ids, ref_ids.to(torch.int32))
    assert torch.equal(vals.view(torch.int16), ref_vals.view(torch.int16))
    q, sf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=256)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(sf.view(m, -1), ref_sf.view(m, -1))


def _quant_input(m: int, seed: int) -> torch.Tensor:
    """BF16 rows over a wide dynamic range, with all-zero blocks, tiny blocks and huge outliers."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, LATENT, generator=g, device="cuda")
    x = x * torch.exp2(
        torch.randint(-12, 12, (m, LATENT // 32, 1), generator=g, device="cuda").float()
    ).repeat(1, 1, 32).view(m, LATENT)
    x[:, :32] = 0
    if m > 1:
        x[1, 64:96] *= 1e-30
        x[-1, 96:128] *= 1e4
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("m", [1, 7, 28, 64, 65, 112, 224, 256])
def test_runner_route_fused_quant(m):
    """The MXFP8 quantization of the routing launch equals ``mxfp8_quantize`` and leaves the routing unchanged."""
    kt = _tail()
    logits, bias = _logits(m, 900 + m)
    x = _quant_input(m, m)
    tile, offset, local = 16, 112, 112
    ids, vals, routed, quantized = kt.run_k3_route_runner(
        logits, m, bias, 1.0, offset, local, NUM_EXPERTS, tile, m <= 64, quant_input=x
    )
    ref_ids, ref_vals, ref_routed, none = kt.run_k3_route_runner(
        logits, m, bias, 1.0, offset, local, NUM_EXPERTS, tile, m <= 64
    )
    assert none is None and quantized is not None
    assert torch.equal(ids, ref_ids) and torch.equal(
        vals.view(torch.int16), ref_vals.view(torch.int16)
    )
    assert torch.equal(routed[0], ref_routed[0]) and torch.equal(routed[5], ref_routed[5])
    q, sf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=256)
    assert torch.equal(quantized[0].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(quantized[1].view(m, -1), sf.view(m, -1))


@pytest.mark.parametrize("m", [65, 112, 196, 256, 257, 448, 1024, 2048])
def test_topk_only_bf16_matches_noaux(m):
    """Top-k only with BF16 weights (the DEP path, KIMI_K31_ROUTER_TAIL_DEP): ids == noaux_tc_op, weights == the
    BF16 cast of its FP32 weights, for the cluster kernel and the multi-CTA kernel."""
    kt = _tail()
    logits, bias = _logits(m, 300 + m)
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.bfloat16, device="cuda")
    ref_vals, ref_ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
    runs = [kt.run_k3_route_wide] + ([kt.run_k3_route_tail] if m <= kt.MAX_M else [])
    for run in runs:
        run(logits, m, bias, 1.0, ids, vals, None, 0, NUM_EXPERTS)
        assert torch.equal(ids, ref_ids.to(torch.int32)), run.__name__
        assert torch.equal(
            vals.view(torch.int16), ref_vals.float().to(torch.bfloat16).view(torch.int16)
        ), run.__name__


def test_cuda_graph_replay():
    kt = _tail()
    m, tile, offset, local = 112, 32, 112, 112
    logits, bias = _logits(m, 5)
    eager = kt.run_k3_route_runner(logits, m, bias, 1.0, offset, local, NUM_EXPERTS, tile, False)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = kt.run_k3_route_runner(logits, m, bias, 1.0, offset, local, NUM_EXPERTS, tile, False)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[0], eager[0])
    assert torch.equal(out[1].view(torch.int16), eager[1].view(torch.int16))
    e2p, _, total, cta_expert, cta_limit, num_ctas = out[2]
    ref_e2p, _, ref_total, ref_expert, ref_limit, ref_num_ctas = eager[2]
    n = int(ref_num_ctas.item())
    assert torch.equal(e2p, ref_e2p)
    assert torch.equal(total, ref_total) and torch.equal(num_ctas, ref_num_ctas)
    assert torch.equal(cta_expert[:n], ref_expert[:n]) and torch.equal(cta_limit[:n], ref_limit[:n])


def _moe_inputs(m: int, local: int, inter: int, seed: int):
    torch.manual_seed(seed)
    dev = "cuda"
    # Random FP4 codes; E8M0 scales near 2^-7 keep FC1 / FC2 finite.
    w13 = torch.randint(0, 256, (local, 2 * inter, LATENT // 2), dtype=torch.uint8, device=dev)
    s13 = torch.randint(118, 122, (local, 2 * inter, LATENT // 32), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (local, LATENT, inter // 2), dtype=torch.uint8, device=dev)
    s2 = torch.randint(118, 122, (local, LATENT, inter // 32), dtype=torch.uint8, device=dev)
    x = torch.randint(0, 256, (m, LATENT), dtype=torch.uint8, device=dev)
    x = x.bitwise_and(0x77).view(torch.float8_e4m3fn)  # finite e4m3 codes
    x_sf = torch.full((m * LATENT // 32,), 127, dtype=torch.uint8, device=dev)
    alpha = torch.full((local,), 4.0 * 25.0, dtype=torch.float32, device=dev)
    beta = torch.full((local,), 0.25, dtype=torch.float32, device=dev)
    return x, x_sf, w13, s13, w2, s2, alpha, beta


def _rows_by_slot(rows: torch.Tensor, e2p: torch.Tensor, m: int):
    idx = e2p.view(-1)[: m * TOP_K].long()
    valid = idx >= 0
    out = torch.zeros(m * TOP_K, rows.shape[1], dtype=rows.dtype, device=rows.device)
    out[valid] = rows[idx[valid]]
    return out, valid


def _has_prerouted(finalized: bool) -> bool:
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import mxfp4_runner_has_prerouted

    return mxfp4_runner_has_prerouted(finalized=finalized)


# (local experts, intermediate size per rank, first local expert): TEP8 EP8, TEP16 EP16, TP4xEP2.
_RUNNER_LAYOUTS = [(112, 3072, 112), (56, 3072, 7 * 56), (448, 768, 448)]


@pytest.mark.parametrize("m", [7, 28, 112, 288])
@pytest.mark.parametrize("local,inter,offset", _RUNNER_LAYOUTS)
def test_prerouted_runner_matches(m, local, inter, offset):
    if not _has_prerouted(False) or not _has_prerouted(True):
        pytest.skip("this build's MXFP4 runner takes no precomputed routing")
    kt = _tail()
    if not kt.runner_sort_fits(m, local):
        pytest.skip("outside the routing tail's tables")
    x, x_sf, w13, s13, w2, s2, alpha, beta = _moe_inputs(m, local, inter, m + offset)
    runner = torch.classes.trtllm.MxE4m3MxE2m1BlockScaleMoERunner(3, True)
    tactics = runner.get_valid_configs(TOP_K, LATENT, inter, local, m, LATENT, inter)
    assert tactics
    seen_tiles = set()
    for tactic in tactics:
        tile = int(tactic[0])
        if tile in seen_tiles:
            continue
        seen_tiles.add(tile)
        _, _, ids, weights, routed = _route(m, 1000 + m + tile, tile, offset, local, m <= 64)
        args = (
            None, None, x, x_sf, w13, s13, None, alpha, beta, None, w2, s2, None, None, None, None,
            NUM_EXPERTS, TOP_K, None, None, inter, LATENT, inter, offset, local, 1.0, 2,
            list(tactic), weights, ids,
        )  # fmt: skip
        ref_rows, _, ref_e2p = runner.run_moe_unfinalized(*args)
        rows, _, _ = runner.run_moe_unfinalized_prerouted(*args, routed, tile)
        ref_out = runner.run_moe(*args, None)
        out = runner.run_moe_prerouted(*args, None, routed, tile)
        torch.cuda.synchronize()
        ref_by_slot, ref_valid = _rows_by_slot(ref_rows, ref_e2p, m)
        by_slot, valid = _rows_by_slot(rows, routed[0], m)
        assert torch.equal(valid, ref_valid)
        assert torch.isfinite(ref_by_slot.float()).all()
        assert torch.equal(by_slot.view(torch.int16), ref_by_slot.view(torch.int16)), tactic
        assert torch.isfinite(ref_out.float()).all()
        assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16)), tactic


@pytest.mark.parametrize("m", [28, 112])
def test_prerouted_ops_fall_back_on_other_tile(m):
    from tensorrt_llm._torch.autotuner import autotune

    if not _has_prerouted(False) or not _has_prerouted(True):
        pytest.skip("this build's MXFP4 runner takes no precomputed routing")
    local, inter, offset = 112, 3072, 0
    x, x_sf, w13, s13, w2, s2, alpha, beta = _moe_inputs(m, local, inter, 7 * m)
    # Untuned: the ops run the default tactic ([-1, -1]), so they route themselves whatever the tile.
    _, _, ids, weights, routed = _route(m, 5 * m, 128, offset, local, m <= 64)
    routed[0].fill_(-7)
    args = (
        None, None, x, x_sf, w13, s13, None, alpha, beta, None, w2, s2, None,
        NUM_EXPERTS, TOP_K, None, None, inter, LATENT, None, offset, local, 1.0, 2, 3,
    )  # fmt: skip
    with autotune(False):
        ref_rows, _, ref_e2p = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner_unfinalized(
            *args, topk_weights=weights, topk_ids=ids
        )
        rows, _ = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner_prerouted(
            *args, weights, ids, routed, 128
        )
        ref_out = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
            *args, topk_weights=weights, topk_ids=ids
        )
        out = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner_prerouted_finalized(
            *args, weights, ids, routed, 128
        )
    ref_by_slot, ref_valid = _rows_by_slot(ref_rows, ref_e2p, m)
    by_slot, valid = _rows_by_slot(rows, routed[0], m)
    assert torch.equal(valid, ref_valid)
    assert torch.equal(by_slot.view(torch.int16), ref_by_slot.view(torch.int16))
    assert torch.equal(out.view(torch.int16), ref_out.view(torch.int16))


@pytest.mark.parametrize("finalized", [False, True])
def test_runner_ops_record_the_tuned_tile(finalized):
    """The plain runner ops record their tuned tile, so the model's next forward can sort for it."""
    from tensorrt_llm._torch.autotuner import autotune
    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as tgo

    m, local, inter, offset = 56, 112, 3072, 0
    x, x_sf, w13, s13, w2, s2, alpha, beta = _moe_inputs(m, local, inter, 3)
    _, _, ids, weights, _ = _route(m, 9, 128, offset, local, True)
    # n_group / topk_group 1: the autotuner's dummy routing runs the DeepSeek-V3 method.
    args = (
        None, None, x, x_sf, w13, s13, None, alpha, beta, None, w2, s2, None,
        NUM_EXPERTS, TOP_K, 1, 1, inter, LATENT, None, offset, local, 1.0, 2, 3,
    )  # fmt: skip
    tgo._MXFP4_PREROUTED_TILE.pop((m, local, TOP_K), None)
    op = (
        tgo.mxe4m3_mxe2m1_block_scale_moe_runner
        if finalized
        else tgo.mxe4m3_mxe2m1_block_scale_moe_runner_unfinalized
    )
    with autotune():
        op(*args, topk_weights=weights, topk_ids=ids)
    assert tgo.mxfp4_prerouted_tile(m, local, TOP_K) in (8, 16, 32, 64, 128, 256)


def _routing_scale_requirements():
    """(name, input_requirement) of a backend reading BF16 routing scales (TRTLLM-Gen) and one reading FP32."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.identity import TRTLLM_GEN_INPUT_REQUIREMENT

    return {
        "trtllm_gen": TRTLLM_GEN_INPUT_REQUIREMENT,
        "fp32_scales": SimpleNamespace(routing_scales_dtype=torch.float32),
    }


@pytest.mark.parametrize("m", [65, 112, 2048])
@pytest.mark.parametrize("backend", ["trtllm_gen", "fp32_scales"])
def test_dep_route_only_for_bf16_routing_scales(monkeypatch, backend, m):
    """KIMI_K31_ROUTER_TAIL_DEP above 64 rows on an attention-DP MoE with a token dispatch: the routing tail hands its
    top-16 (BF16 weights) to the routing method only for a backend that reads BF16 routing scales (TRTLLM-Gen, where
    it equals the scheduler's cast of noaux_tc's weights). A backend asking for FP32 scales keeps noaux_tc's FP32
    routing: the tail stays off and nothing is precomputed."""
    from tensorrt_llm._torch.models import modeling_kimi_linear as mkl

    requirement = _routing_scale_requirements()[backend]
    monkeypatch.setattr(mkl, "_KimiK3RoutingMethod", SimpleNamespace)
    logits, bias = _logits(m, 900 + m)
    routing = SimpleNamespace(precomputed=None)
    moe = SimpleNamespace(
        backend=SimpleNamespace(input_requirement=requirement),
        use_dp=True,
        comm=object(),
        apply_router_weight_on_input=False,
        routing_method=routing,
        calculate_num_chunks=lambda rank_tokens: 1,
    )
    rt = SimpleNamespace(
        routed_experts=moe,
        _k31_router_tail_dep_wanted=True,
        expert_replicas=None,
        top_k=TOP_K,
        gate=SimpleNamespace(
            num_experts=NUM_EXPERTS, e_score_correction_bias=bias, routed_scaling_factor=1.0
        ),
    )
    took = mkl.KimiK3MoERuntime._k31_dep_route(rt, logits, [m, m])
    bf16_scales = requirement.routing_scales_dtype == torch.bfloat16
    assert took is bf16_scales and mkl._k31_takes_bf16_routing_scales(moe.backend) is bf16_scales
    if not bf16_scales:
        assert routing.precomputed is None
        return
    ids, vals = routing.precomputed
    ref_vals, ref_ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, 1.0)
    assert torch.equal(ids, ref_ids.to(torch.int32))
    assert torch.equal(
        vals.view(torch.int16), ref_vals.float().to(torch.bfloat16).view(torch.int16)
    )
