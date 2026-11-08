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
"""The decode routing kernel (``kimi_k3_routing``: ungrouped ``noaux_tc`` top-16 + ``moe_sort`` in
one launch, ``trtllm::noaux_tc_route_sort``) against ``noaux_tc_op`` -> ``moe_sort`` on the same
FP32 logits.

* Top-k ids and weights must be bit-identical to ``noaux_tc_op``.
* The ``moe_sort`` outputs must be bit-identical to a token-ordered sort of the ids and match
  ``moe_sort``'s tile metadata exactly (``moe_sort``'s cluster kernel orders the rows of a tile
  by atomics above 12 tokens, so there only the set of rows per expert is compared).
"""

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import (
    IS_CUTLASS_DSL_AVAILABLE,
    IS_CUTLASS_DSL_RUBIN_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not IS_CUTLASS_DSL_AVAILABLE
    or not torch.cuda.is_available()
    or not (
        torch.cuda.get_device_capability() in ((10, 0), (10, 3))
        or (torch.cuda.get_device_capability() == (10, 7) and IS_CUTLASS_DSL_RUBIN_AVAILABLE)
    ),
    reason="needs an SM100/SM103 GPU, or SM107 with a Rubin-capable CuTe DSL",
)

NUM_EXPERTS = 896
TOP_K = 16
TILE = 128
MS = [1, 4, 5, 7, 13, 16, 17, 28, 32, 33, 56, 64, 65, 112, 128, 129, 138, 224, 256]
# (local_expert_offset, local_num_experts): EP16 ranks, EP8, TP4xEP4 ranks, TP8xEP2, no EP.
LAYOUTS = [
    (0, 56),
    (7 * 56, 56),
    (15 * 56, 56),
    (448, 112),
    (0, 224),
    (3 * 224, 224),
    (448, 448),
    (0, 896),
]


def _module():
    import tensorrt_llm  # noqa: F401  (registers the reference ops)
    from tensorrt_llm._torch.moe.kernels.blackwell import kimi_k3_routing

    return kimi_k3_routing


def _logits(m, seed, mode="rand", num_experts=NUM_EXPERTS):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    if mode == "realistic":
        # Router-like logits: hidden states (unit norm) against unit expert directions, plus
        # per-expert popularity; the tokens of one request (an MTP verify window of 7) share most
        # of their hidden state; bias spread ~0.19 as in Kimi-K3.
        dim = 512
        directions = torch.randn((NUM_EXPERTS, dim), device="cuda", generator=gen)
        directions = directions / directions.norm(dim=1, keepdim=True)
        popularity = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.7
        requests = torch.randn(((m + 6) // 7, dim), device="cuda", generator=gen)
        hidden = requests.repeat_interleave(7, dim=0)[:m] * 0.85
        hidden = hidden + torch.randn((m, dim), device="cuda", generator=gen) * 0.5
        hidden = hidden / hidden.norm(dim=1, keepdim=True)
        logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
        bias = (torch.rand((NUM_EXPERTS,), device="cuda", generator=gen) - 0.5) * 0.19
        return logits.contiguous(), bias
    if mode == "rand":
        logits = torch.randn((m, num_experts), device="cuda", generator=gen) * 2
        bias = torch.randn((num_experts,), device="cuda", generator=gen) * 0.01
    elif mode == "skew":
        logits = torch.randn((m, NUM_EXPERTS), device="cuda", generator=gen)
        logits[:, 224:232] += 6
        bias = torch.randn((NUM_EXPERTS,), device="cuda", generator=gen) * 0.01
    else:  # ties: all-equal rows and rows with few distinct values (exact arg-max fallback)
        logits = torch.zeros((m, num_experts), device="cuda")
        bias = torch.zeros((num_experts,), device="cuda")
        half = m // 2
        logits[:half] = torch.randint(-2, 3, (half, num_experts), device="cuda", generator=gen)
    return logits, bias


def _route(logits, bias, scale, offset, num_local):
    rt = _module()
    m = logits.shape[0]
    assert rt.can_run(logits, bias, offset, num_local)
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.float32, device="cuda")
    sort = rt.sort_outputs(m, num_local, logits.device)
    rt.run_route_sort(logits, bias, scale, offset, num_local, ids, vals, sort)
    return (ids, vals, *sort)


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
    ids, vals, tile_expert, tile_limit, e2p, p2e, total, num_tiles = outs
    ref_ids, ref_vals = _noaux(logits, bias, scale)
    _assert_bit_equal(ids, ref_ids, "topk_ids")
    _assert_bit_equal(vals, ref_vals, "topk_weights")

    m = ids.shape[0]
    ref = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=vals,
        num_experts=logits.shape[1],
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
@pytest.mark.parametrize("mode", ["realistic", "rand", "skew", "ties"])
def test_route_sort_matches_noaux_and_moe_sort(m, mode):
    logits, bias = _logits(m, seed=m, mode=mode)
    for offset, num_local in LAYOUTS:
        outs = _route(logits, bias, 1.0, offset, num_local)
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


@pytest.mark.parametrize("m", [7, 32, 224])
@pytest.mark.parametrize("scale", [2.5, 0.3])
def test_route_sort_routed_scaling_factor(m, scale):
    logits, bias = _logits(m, seed=100 + m)
    outs = _route(logits, bias, scale, 224, 224)
    _check_outputs(outs, logits, bias, scale, 224, 224)


@pytest.mark.parametrize("num_experts", [128, 256, 384, 512, 1024])
@pytest.mark.parametrize("m", [1, 32, 33, 64, 65, 128, 129, 256])  # each launch config
@pytest.mark.parametrize("mode", ["rand", "ties"])
def test_route_sort_other_expert_counts(num_experts, m, mode):
    """Expert counts other than Kimi-K3's 896: every multiple of 128 up to 1024 is served."""
    logits, bias = _logits(m, seed=3 * m + num_experts, mode=mode, num_experts=num_experts)
    quarter = num_experts // 4
    for offset, num_local in ((0, num_experts), (quarter, quarter), (num_experts - 16, 16)):
        outs = _route(logits, bias, 1.0, offset, num_local)
        _check_outputs(outs, logits, bias, 1.0, offset, num_local)


def test_route_sort_cuda_graph_replay():
    rt = _module()
    m = 32
    logits, bias = _logits(m, seed=3)
    static = logits.clone()
    _route(static, bias, 1.0, 224, 224)
    ids = torch.empty((m, TOP_K), dtype=torch.int32, device="cuda")
    vals = torch.empty((m, TOP_K), dtype=torch.float32, device="cuda")
    sort = rt.sort_outputs(m, 224, static.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rt.run_route_sort(static, bias, 1.0, 224, 224, ids, vals, sort)
    for seed in range(4):
        fresh, _ = _logits(m, seed=10 + seed, mode="skew" if seed % 2 else "rand")
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        _check_outputs((ids, vals, *sort), static, bias, 1.0, 224, 224)


def test_route_sort_op():
    if not hasattr(torch.ops.trtllm, "noaux_tc_route_sort"):
        pytest.skip("trtllm::noaux_tc_route_sort is not registered")
    m = 64
    logits, bias = _logits(m, seed=5, mode="realistic")
    outs = torch.ops.trtllm.noaux_tc_route_sort(logits, bias, 2.5, 7 * 56, 56)
    _check_outputs(outs, logits, bias, 2.5, 7 * 56, 56)


def test_can_run():
    rt = _module()
    logits, bias = _logits(8, seed=1)
    assert rt.can_run(logits, bias, 0, 56)
    assert rt.can_run(logits, bias, NUM_EXPERTS - 56, 56)
    assert not rt.can_run(logits, bias, NUM_EXPERTS - 55, 56)
    assert not rt.can_run(logits, bias, -1, 56)
    assert not rt.can_run(logits, bias, 0, 0)
    assert not rt.can_run(logits, bias, 0, NUM_EXPERTS + 1)
    assert not rt.can_run(logits.double(), bias, 0, 56)
    assert not rt.can_run(logits[:, :-1], bias[:-1], 0, 56)
    assert not rt.can_run(logits[:, :-1].contiguous(), bias[:-1].contiguous(), 0, 56)
    assert not rt.can_run(logits, bias[:-1], 0, 56)
    assert not rt.can_run(logits.t().contiguous().t(), bias, 0, 56)
    assert not rt.can_run(logits.cpu(), bias.cpu(), 0, 56)
    too_many, too_many_bias = _logits(rt.MAX_M + 1, seed=1)
    assert not rt.can_run(too_many, too_many_bias, 0, 56)
    wide, wide_bias = _logits(8, seed=1, num_experts=rt.MAX_EXPERTS + 128)
    assert not rt.can_run(wide, wide_bias, 0, 56)
    assert rt.serves(8, NUM_EXPERTS, 0, 56, logits.device)
    assert not rt.serves(8, 960, 0, 56, logits.device)


@pytest.mark.parametrize("rubin_dsl", [False, True], ids=["stock_dsl", "rubin_dsl"])
def test_sm107_selected_only_with_rubin_cute_dsl(monkeypatch, rubin_dsl):
    """On SM107 the kernel serves only with a Rubin-capable CuTe DSL; otherwise
    ``apply_with_sort`` declines and routing stays on ``noaux_tc_op`` + ``moe_sort``."""
    from tensorrt_llm._torch import cute_dsl_utils
    from tensorrt_llm._torch.moe.fused_moe.routing import DeepSeekV3MoeRoutingMethod

    rt = _module()
    logits, bias = _logits(8, seed=1)
    monkeypatch.setattr(rt, "_device_capability", lambda device_index: (10, 7))
    monkeypatch.setattr(cute_dsl_utils, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", rubin_dsl)
    assert rt.can_run(logits, bias, 0, 56) == rubin_dsl
    assert rt.serves(8, NUM_EXPERTS, 0, 56, logits.device) == rubin_dsl
    routing = DeepSeekV3MoeRoutingMethod(
        top_k=TOP_K,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        callable_e_score_correction_bias=lambda: bias,
        is_fused=True,
    )
    assert routing.serves_with_sort(8, NUM_EXPERTS, 0, 56, logits.device) == rubin_dsl
    assert (routing.apply_with_sort(logits, 0, 56) is not None) == rubin_dsl


def test_route_sort_op_rejects_unsupported_operands():
    """``trtllm::noaux_tc_route_sort`` raises on operands the kernel does not serve, so a
    direct caller cannot run past the kernel's token limit into its fixed buffers."""
    m_over = 257
    logits = torch.zeros((m_over, NUM_EXPERTS), dtype=torch.float32, device="cuda")
    bias = torch.zeros((NUM_EXPERTS,), dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="256"):
        torch.ops.trtllm.noaux_tc_route_sort(logits, bias, 1.0, 0, NUM_EXPERTS)
    with pytest.raises(ValueError, match="dtype=torch.bfloat16"):
        torch.ops.trtllm.noaux_tc_route_sort(logits.to(torch.bfloat16), bias, 1.0, 0, NUM_EXPERTS)


def test_route_sort_op_serves_valid_logits():
    """A supported call through the registered op server matches the kernel API."""
    logits, bias = _logits(4, seed=3)
    out = torch.ops.trtllm.noaux_tc_route_sort(logits, bias, 1.5, 0, NUM_EXPERTS)
    ref_ids, _ = _noaux(logits, bias, 1.5)
    _assert_bit_equal(out[0], ref_ids, "op ids")
