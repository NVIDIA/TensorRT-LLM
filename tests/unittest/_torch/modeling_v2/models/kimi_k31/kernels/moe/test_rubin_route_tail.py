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
"""Correctness of the Kimi K3.1 routing tail (``rubin_route_tail.k3_route_tail``).

``k3_route_tail`` replaces ``noaux_tc_op`` -> ``moe_sort`` on given FP32 logits (one cluster up to
256 tokens, a grid of CTAs with a last-CTA sort above):

* top-k ids and weights must be bit-identical to ``noaux_tc_op`` on the same logits;
* the ``moe_sort`` outputs must be bit-identical to a token-ordered sort of the ids, and match
  ``moe_sort``'s tile metadata exactly (``moe_sort``'s cluster kernel orders the rows of a tile
  by atomics above 12 tokens, so there only the set of rows per tile is compared).
"""

import pytest
import torch
from _rubin_route_tail_inputs import router_logits

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="needs an SM107 GPU",
)

NUM_EXPERTS = 896
TOP_K = 16
TILE = 128
MS = [1, 4, 5, 7, 13, 16, 17, 28, 32, 33, 56, 64, 65, 112, 128, 224, 256]
# (local_expert_offset, local_num_experts): DEP16's EP16 ranks (56), TEP8's EP8 ranks (112),
# helix8's TP2xEP4 ranks (224), and wider windows.
LAYOUTS = [
    (0, 56),
    (7 * 56, 56),
    (15 * 56, 56),
    (3 * 112, 112),
    (7 * 112, 112),
    (0, 224),
    (3 * 224, 224),
    (448, 448),
    (0, 896),
]


def _module():
    import tensorrt_llm  # noqa: F401  (registers the reference ops)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_route_tail,
    )

    return rubin_route_tail


def _noaux(logits, bias, scale):
    vals, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, scale)
    return ids.to(torch.int32), vals.to(torch.float32)


def _assert_bit_equal(actual, expected, what):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, what
    a = actual.view(torch.int32) if actual.dtype == torch.float32 else actual
    e = expected.view(torch.int32) if expected.dtype == torch.float32 else expected
    mismatch = (a != e).nonzero()
    assert mismatch.numel() == 0, f"{what}: {mismatch.shape[0]} mismatches, first {mismatch[:4]}"


def _token_ordered_sort(ids, offset, num_local):
    """Python ``moe_sort`` with the rows of each expert in ascending token order."""
    m = ids.shape[0]
    rows = {}
    for t, row in enumerate(ids.cpu().tolist()):
        for k, e in enumerate(row):
            j = e - offset
            if 0 <= j < num_local:
                rows.setdefault(j, []).append(t * TOP_K + k)
    tile_expert, tile_limit, e2p, p2e = [], [], torch.full((m, TOP_K), -1, dtype=torch.int32), {}
    for j in sorted(rows):
        base = len(tile_expert) * TILE
        count = len(rows[j])
        for c in range((count + TILE - 1) // TILE):
            tile_expert.append(j)
            tile_limit.append(min(base + (c + 1) * TILE, base + count))
        for r, expanded in enumerate(rows[j]):
            e2p.view(-1)[expanded] = base + r
            p2e[base + r] = expanded
    return tile_expert, tile_limit, e2p, p2e


def _check_outputs(outs, logits, bias, scale, offset, num_local):
    ids, vals, tile_expert, tile_limit, e2p, p2e, total, num_tiles = outs[:8]
    ref_ids, ref_vals = _noaux(logits, bias, scale)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")

    m = ids.shape[0]
    ref = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=vals,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=offset,
        local_num_experts=num_local,
        tile_tokens_dim=TILE,
    )
    for got, exp in zip((tile_expert, tile_limit, e2p, p2e, total, num_tiles), ref):
        assert got.dtype == exp.dtype == torch.int32 and got.shape == exp.shape
    n = int(num_tiles.item())
    _assert_bit_equal(num_tiles, ref[5], "num_non_exiting_tiles")
    _assert_bit_equal(total, ref[4], "total_num_padded_tokens")
    _assert_bit_equal(tile_expert[:n], ref[0][:n], "tile_idx_to_expert_idx")
    _assert_bit_equal(tile_limit[:n], ref[1][:n], "tile_idx_to_mn_limit")

    c_expert, c_limit, c_e2p, c_p2e = _token_ordered_sort(ids, offset, num_local)
    assert tile_expert[:n].tolist() == c_expert and tile_limit[:n].tolist() == c_limit
    _assert_bit_equal(e2p.cpu(), c_e2p, "expanded_idx_to_permuted_idx (token order)")
    p2e_cpu = p2e.cpu()
    for perm, expanded in c_p2e.items():
        assert int(p2e_cpu[perm]) == expanded, f"permuted_idx_to_expanded_idx[{perm}]"

    ref_e2p, ref_p2e = ref[2].cpu(), ref[3].cpu()
    if m <= 12:
        _assert_bit_equal(e2p.cpu(), ref_e2p, "expanded_idx_to_permuted_idx vs moe_sort")
    else:
        _assert_bit_equal(e2p.cpu() < 0, ref_e2p < 0, "non-local entries vs moe_sort")
    # moe_sort's cluster kernel also splits a multi-tile expert's rows over its tiles in atomic
    # order, so the rows are compared per expert.
    rows, ref_rows = {}, {}
    for t in range(n):
        lo, hi = t * TILE, int(tile_limit[t])
        expert = int(tile_expert[t])
        rows.setdefault(expert, []).extend(p2e_cpu[lo:hi].tolist())
        ref_rows.setdefault(expert, []).extend(ref_p2e[lo:hi].tolist())
    for expert in rows:
        assert sorted(rows[expert]) == sorted(ref_rows[expert]), f"local expert {expert}"


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("mode", ["rand", "skew", "ties"])
def test_tail_matches_noaux_and_sort(m, mode):
    rt = _module()
    logits, bias = router_logits(m, seed=m, mode=mode)
    for offset, num_local in LAYOUTS:
        outs = rt.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE)
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [7, 32, 224])
@pytest.mark.parametrize("scale", [2.5, 0.3])
def test_tail_routed_scaling_factor(m, scale):
    rt = _module()
    logits, bias = router_logits(m, seed=100 + m)
    outs = rt.k3_route_tail(logits, bias, scale, 224, 224, TILE)
    _check_outputs(outs, logits, bias, scale, 224, 224)


@pytest.mark.parametrize("m", [1, 7, 16, 32, 64, 128])
@pytest.mark.parametrize(
    "num_ctas,warps_per_token,num_warps",
    [(1, 1, 16), (2, 2, 16), (4, 4, 16), (8, 1, 16), (8, 1, 8), (8, 2, 8), (4, 4, 8)],
)
def test_tail_configs(m, num_ctas, warps_per_token, num_warps):
    rt = _module()
    if (m + num_ctas - 1) // num_ctas > 32:
        pytest.skip("more than 32 tokens per CTA")
    logits, bias = router_logits(m, seed=7 * m + num_ctas)
    outs = rt._route_tail_outputs(logits, m, 56, TILE)
    rt.run_k3_route_tail(
        logits,
        m,
        bias,
        1.0,
        outs[0],
        outs[1],
        outs[2:],
        7 * 56,
        56,
        num_ctas=num_ctas,
        warps_per_token=warps_per_token,
        num_warps=num_warps,
    )
    _check_outputs(outs, logits, bias, 1.0, 7 * 56, 56)


@pytest.mark.parametrize("m", [1, 7, 32, 64, 96, 128, 256])
@pytest.mark.parametrize("mode", ["realistic", "rand", "skew", "ties"])
def test_tail_approx_candidates(m, mode):
    """``route_approx``: candidates from ``tanh.approx`` scores with a margin,
    the exact sigmoid only for them. The outputs stay bit-identical to ``noaux_tc_op`` and the
    token-ordered sort, also for ties (candidate overflow: exact fallback), split partials and
    other configs."""
    rt = _module()
    logits, bias = router_logits(m, seed=31 * m + 1, mode=mode)
    for offset, num_local in LAYOUTS[::2]:
        outs = rt.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE, route_approx=True)
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)
    for num_ctas, warps_per_token, num_warps in ((8, 2, 16), (8, 1, 12), (4, 4, 16)):
        if (m + num_ctas - 1) // num_ctas > 32:
            continue
        outs = rt._route_tail_outputs(logits, m, 56, TILE)
        rt.run_k3_route_tail(
            logits,
            m,
            bias,
            2.5,
            outs[0],
            outs[1],
            outs[2:],
            7 * 56,
            56,
            num_ctas=num_ctas,
            warps_per_token=warps_per_token,
            num_warps=num_warps,
            route_approx=True,
        )
        _check_outputs(outs, logits, bias, 2.5, 7 * 56, 56)
    gen = torch.Generator(device="cuda").manual_seed(m)
    partials = torch.randn((3, m + 1, NUM_EXPERTS), device="cuda", generator=gen)
    outs = rt.k3_route_tail(partials, bias, 1.0, 0, 224, TILE, m, route_approx=True)
    summed = partials[0, :m] + partials[1, :m] + partials[2, :m]
    _check_outputs(outs, summed, bias, 1.0, 0, 224)


@pytest.mark.parametrize("m", [7, 32, 100])
def test_tail_split_partials(m):
    """``[S, R, 896]`` partials (R > M) are summed in split order."""
    rt = _module()
    gen = torch.Generator(device="cuda").manual_seed(m)
    partials = torch.randn((5, m + 3, NUM_EXPERTS), device="cuda", generator=gen)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    outs = rt.k3_route_tail(partials, bias, 1.0, 0, 224, TILE, m)
    logits = partials[0, :m].clone()
    for s in range(1, 5):
        logits = logits + partials[s, :m]
    _check_outputs(outs, logits, bias, 1.0, 0, 224)


def test_tail_cuda_graph_replay():
    rt = _module()
    m = 32
    logits, bias = router_logits(m, seed=3)
    static = logits.clone()
    rt.k3_route_tail(static, bias, 1.0, 224, 224, TILE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = rt.k3_route_tail(static, bias, 1.0, 224, 224, TILE)
    for seed in range(4):
        fresh, _ = router_logits(m, seed=10 + seed, mode="skew" if seed % 2 else "rand")
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        _check_outputs(outs, static, bias, 1.0, 224, 224)


# Multi-CTA path (M > 256)

WIDE_MS = [257, 384, 512, 896, 1344]
# DEP16's EP16 ranks (56 local experts), TEP8's EP8 (112) and helix8's TP2xEP4 (224).
WIDE_LAYOUTS = [(0, 56), (7 * 56, 56), (15 * 56, 56), (448, 112), (3 * 224, 224)]


@pytest.mark.parametrize("m", WIDE_MS)
@pytest.mark.parametrize("mode", ["realistic", "rand", "skew", "ties"])
def test_wide_matches_noaux_and_sort(m, mode):
    rt = _module()
    logits, bias = router_logits(m, seed=m, mode=mode)
    for offset, num_local in WIDE_LAYOUTS:
        if not rt.wide_sort_fits(m, num_local):
            assert not rt.wide_table_fits(m, num_local, 8)
            assert not rt.wide_table_fits(m, num_local, 16)
            with pytest.raises(ValueError):
                rt.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE)
            continue
        outs = rt.k3_route_tail(logits, bias, 1.0, offset, num_local, TILE)
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [257, 896, 2048])
def test_wide_routed_scaling_factor_and_max_m(m):
    rt = _module()
    logits, bias = router_logits(m, seed=200 + m, mode="realistic")
    outs = rt.k3_route_tail(logits, bias, 2.5, 7 * 56, 56, TILE)
    _check_outputs(outs, logits, bias, 2.5, 7 * 56, 56)


@pytest.mark.parametrize("m", [300, 896, 1344])
@pytest.mark.parametrize(
    "tokens_per_cta,num_warps,warps_per_token",
    [(8, 8, 1), (8, 16, 1), (16, 8, 1), (16, 16, 1), (16, 8, 2), (16, 16, 4)],
)
def test_wide_configs(m, tokens_per_cta, num_warps, warps_per_token):
    rt = _module()
    logits, bias = router_logits(m, seed=11 * m + tokens_per_cta, mode="realistic")
    for offset, num_local in [(7 * 56, 56), (224, 224)]:
        outs = rt._route_tail_outputs(logits, m, num_local, TILE)
        if not rt.wide_table_fits(m, num_local, tokens_per_cta):
            with pytest.raises(ValueError):
                rt.run_k3_route_wide(
                    logits,
                    m,
                    bias,
                    1.0,
                    outs[0],
                    outs[1],
                    outs[2:],
                    offset,
                    num_local,
                    tokens_per_cta=tokens_per_cta,
                )
            continue
        # Poison the outputs: every entry the contract covers must be written.
        for t in outs:
            t.view(torch.int32).fill_(-7)
        rt.run_k3_route_wide(
            logits,
            m,
            bias,
            1.0,
            outs[0],
            outs[1],
            outs[2:],
            offset,
            num_local,
            tokens_per_cta=tokens_per_cta,
            num_warps=num_warps,
            warps_per_token=warps_per_token,
        )
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [257, 1344, 2048])
@pytest.mark.parametrize("tokens_per_cta,num_warps", [(8, 4), (16, 8), (32, 8)])
def test_wide_topk_only(m, tokens_per_cta, num_warps):
    rt = _module()
    logits, bias = router_logits(m, seed=m + 1, mode="realistic")
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.float32, device="cuda")
    rt.run_k3_route_wide(
        logits,
        m,
        bias,
        1.0,
        ids,
        vals,
        None,
        0,
        NUM_EXPERTS,
        tokens_per_cta=tokens_per_cta,
        num_warps=num_warps,
    )
    ref_ids, ref_vals = _noaux(logits, bias, 1.0)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")


@pytest.mark.parametrize("m", [2049, 16384, 65536])
def test_wide_topk_only_past_the_sorted_bound(m):
    """A top-k-only launch (independent CTAs, no workspace) takes up to WIDE_TOPK_MAX_M tokens:
    a context rank's whole chunk; with the sort the bound stays WIDE_MAX_M."""
    rt = _module()
    assert rt.WIDE_MAX_M < m <= rt.WIDE_TOPK_MAX_M
    logits, bias = router_logits(m, seed=3 * m, mode="realistic")
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.float32, device="cuda")
    rt.run_k3_route_wide(logits, m, bias, 1.0, ids, vals, None, 0, NUM_EXPERTS)
    ref_ids, ref_vals = _noaux(logits, bias, 1.0)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")
    if m == 2049:
        outs = rt._route_tail_outputs(logits, m, 56, TILE)
        with pytest.raises(ValueError):
            rt.run_k3_route_wide(logits, m, bias, 1.0, outs[0], outs[1], outs[2:], 0, 56)


def test_wide_topk_only_refuses_past_its_bound():
    rt = _module()
    m = rt.WIDE_TOPK_MAX_M + 1
    logits = torch.zeros((m, NUM_EXPERTS), device="cuda")
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError):
        rt.run_k3_route_wide(
            logits, m, torch.zeros(NUM_EXPERTS, device="cuda"), 1.0, ids, vals, None, 0, 896
        )


def test_wide_split_partials():
    """``[S, R, 896]`` partials (R > M) are summed in split order above 256 tokens too."""
    rt = _module()
    m = 300
    gen = torch.Generator(device="cuda").manual_seed(m)
    partials = torch.randn((2, m + 5, NUM_EXPERTS), device="cuda", generator=gen)
    bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    outs = rt.k3_route_tail(partials, bias, 1.0, 7 * 56, 56, TILE, m)
    _check_outputs(outs, partials[0, :m] + partials[1, :m], bias, 1.0, 7 * 56, 56)


def test_wide_default_config():
    """8-token CTAs (compact sort tables) up to 1024 tokens where the tables fit, else 16."""
    rt = _module()
    assert rt.default_wide_config(384, 56) == (8, 8, 1)
    assert rt.default_wide_config(896, 56) == (8, 8, 1)
    assert rt.default_wide_config(1344, 56) == (16, 16, 1)
    assert rt.wide_table_fits(1344, 56, 8)
    assert rt.default_wide_config(2048, 56) == (16, 16, 1)
    assert rt.wide_sort_fits(2048, 56)
    assert rt.default_wide_config(512, 224) == (16, 16, 1)


def test_wide_rejects_large_local_range():
    rt = _module()
    logits, bias = router_logits(300, seed=1)
    assert not rt.wide_sort_fits(300, 448)
    assert rt.wide_sort_fits(300, 224)
    assert not rt.wide_sort_fits(1344, 224)
    with pytest.raises(ValueError):
        rt.k3_route_tail(logits, bias, 1.0, 0, 448, TILE)


def test_wide_cuda_graph_replay():
    """Replays reuse the arrival counter: several launches (different grids) per graph."""
    rt = _module()
    shapes = [(896, 7 * 56, 56), (512, 224, 224), (1344, 15 * 56, 56)]
    static_inputs, outs = [], []
    for m, offset, num_local in shapes:
        logits, bias = router_logits(m, seed=3 + m, mode="realistic")
        static_inputs.append((logits.clone(), bias))
        rt.k3_route_tail(static_inputs[-1][0], bias, 1.0, offset, num_local, TILE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for (static, bias), (m, offset, num_local) in zip(static_inputs, shapes):
            outs.append(rt.k3_route_tail(static, bias, 1.0, offset, num_local, TILE))
    for seed in range(4):
        for (static, bias), (m, _, _) in zip(static_inputs, shapes):
            mode = ("realistic", "rand", "skew", "ties")[seed]
            static.copy_(router_logits(m, seed=20 + seed, mode=mode)[0])
        graph.replay()
        torch.cuda.synchronize()
        for (static, bias), out, (m, offset, num_local) in zip(static_inputs, outs, shapes):
            _check_outputs(out, static, bias, 1.0, offset, num_local)
