# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_route_tail entry: every cell against the plain-torch reference on SM107, and the
reference itself against scalar models on the CPU. (The kernel's own test is
kernels/moe/test_rubin_route_tail.py, hence this file's name.)"""

import struct
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (
    NUM_EXPERTS,
    TOP_K,
    noaux_topk,
    sort_buffer_sizes,
    token_ordered_sort,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_route_tail import (  # noqa: E501
    rubin_route_tail,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
# Certified on SM107 only; other GPUs skip the file, a CPU-only host runs the cpu_only tests.
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")


def _inputs(rows, seed, mode="realistic", device="cpu"):
    """Router-like logits (requests of 8 rows share most of their hidden state, hot experts) and a
    bias spread of ~0.19, as K3.1's; or tied scores."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "ties":
        logits = torch.zeros(rows, NUM_EXPERTS)
        logits[: rows // 2] = torch.randint(-2, 3, (rows // 2, NUM_EXPERTS), generator=g).float()
        return logits.to(device), torch.zeros(NUM_EXPERTS, device=device)
    dim = 512
    directions = torch.randn(NUM_EXPERTS, dim, generator=g)
    directions /= directions.norm(dim=1, keepdim=True)
    popularity = torch.randn(NUM_EXPERTS, generator=g) * 0.7
    hidden = torch.randn((rows + 7) // 8, dim, generator=g).repeat_interleave(8, 0)[:rows] * 0.85
    hidden = hidden + torch.randn(rows, dim, generator=g) * 0.5
    hidden /= hidden.norm(dim=1, keepdim=True)
    logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
    bias = (torch.rand(NUM_EXPERTS, generator=g) - 0.5) * 0.19
    return logits.contiguous().to(device), bias.to(device)


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _scalar_topk(logits, bias, scale):
    """noaux_tc one row at a time in Python floats on the reference's sigmoid values: selection
    ties to the lower expert, FP32 sum in pick order, FP64 renormalization rounded to FP32."""
    ids, weights = [], []
    for row in logits:
        scores = (0.5 * torch.tanh(0.5 * row) + 0.5).tolist()
        choice = [_f32(s + b) for s, b in zip(scores, bias.tolist())]
        pick = sorted(range(NUM_EXPERTS), key=lambda e: (-choice[e], e))[:TOP_K]
        total = 0.0
        for e in pick:
            total = _f32(total + scores[e])
        ids.append(pick)
        weights.append([_f32(scores[e] * scale / (total + 1e-20)) for e in pick])
    return ids, weights


def _dict_sort(ids, offset, num_local, tile=128):
    """moe_sort with token-ordered rows, as dictionaries (an independent model)."""
    rows = {}
    for t, row in enumerate(ids):
        for k, e in enumerate(row):
            if 0 <= e - offset < num_local:
                rows.setdefault(e - offset, []).append(t * TOP_K + k)
    tile_expert, tile_limit, e2p, p2e = [], [], {}, {}
    for j in sorted(rows):
        base = len(tile_expert) * tile
        for c in range((len(rows[j]) + tile - 1) // tile):
            tile_expert.append(j)
            tile_limit.append(min(base + (c + 1) * tile, base + len(rows[j])))
        for r, expanded in enumerate(rows[j]):
            e2p[expanded] = base + r
            p2e[base + r] = expanded
    return tile_expert, tile_limit, e2p, p2e


def _check(out, ref):
    ids, weights, tile_expert, tile_limit, e2p, p2e, total, num_tiles = out
    r_ids, r_weights, r_expert, r_limit, r_e2p, r_p2e, r_total, r_tiles = ref
    assert torch.equal(ids, r_ids), "selected experts differ"
    # The FP32 total of the 16 picks is a reduction whose order the reference does not model:
    # the renormalized weights may sit two ulp apart. The kernel's bits are noaux_tc_op's
    # (test_certified_cells checks that exactly).
    torch.testing.assert_close(weights, r_weights, rtol=2**-22, atol=0)
    n = int(r_tiles.item())
    assert int(num_tiles.item()) == n and int(total.item()) == int(r_total.item())
    assert torch.equal(tile_expert[:n], r_expert[:n]) and torch.equal(tile_limit[:n], r_limit[:n])
    assert torch.equal(e2p, r_e2p), "expanded_idx_to_permuted_idx differs"
    valid = r_p2e >= 0
    assert torch.equal(p2e[valid], r_p2e[valid]), "permuted_idx_to_expanded_idx differs"


def _kwargs(spec):
    return dict(
        route_approx=spec.get("route_approx", False),
        route_tail_bands=spec.get("route_tail_bands", ""),
    )


@gpu
@pytest.mark.parametrize(
    "cell", rubin_route_tail.CELLS, ids=[c.why[:48] for c in rubin_route_tail.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_route_tail)
    spec = cell.spec
    logits, bias = _inputs(
        spec["rows"], zlib.crc32(cell.why.encode()), spec.get("mode", "realistic"), "cuda"
    )
    args = (logits, bias, spec.get("scale", 1.0), spec["offset"], spec["local"])
    out = rubin_route_tail(*args, **_kwargs(spec))
    _check(out, rubin_route_tail.reference(*args, **_kwargs(spec)))
    # The kernel's own claim: noaux_tc_op's bits.
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers noaux_tc_op

    weights, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, spec.get("scale", 1.0))
    assert torch.equal(out[0], ids.to(torch.int32))
    assert torch.equal(out[1].view(torch.int32), weights.float().view(torch.int32))


@gpu
@pytest.mark.parametrize("rows", [8, 96, 512])
def test_cuda_graph_replay_matches_reference(rows):
    """A launch captured after one eager call (which also creates the multi-CTA workspace)
    replays on new logits."""
    logits, bias = _inputs(rows, 11, device="cuda")
    rubin_route_tail(logits, bias, 1.0, 0, 112)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rubin_route_tail(logits, bias, 1.0, 0, 112)
    for seed in (12, 13):
        logits.copy_(_inputs(rows, seed, device="cuda")[0])
        graph.replay()
        torch.cuda.synchronize()
        _check(out, rubin_route_tail.reference(logits, bias, 1.0, 0, 112))


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_reference_topk_matches_a_scalar_model(scale):
    logits, bias = _inputs(3, 5)
    ids, weights = noaux_topk(logits, bias, scale)
    s_ids, s_weights = _scalar_topk(logits, bias, scale)
    assert ids.tolist() == s_ids
    # The FP32 sum of the 16 picks may take another order than torch's: two ulp at most.
    torch.testing.assert_close(weights, torch.tensor(s_weights), rtol=2**-22, atol=0)


@pytest.mark.cpu_only
def test_reference_ties_go_to_the_lower_expert():
    logits, bias = torch.zeros(2, NUM_EXPERTS), torch.zeros(NUM_EXPERTS)
    ids, weights = noaux_topk(logits, bias, 1.0)
    assert ids.tolist() == [list(range(TOP_K))] * 2
    assert torch.equal(weights, torch.full((2, TOP_K), 1.0 / TOP_K))


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 7, 33, 200])
@pytest.mark.parametrize("offset, local", [(0, 112), (784, 112), (224, 224), (840, 56), (0, 896)])
def test_reference_sort_matches_a_dict_model(rows, offset, local):
    logits, bias = _inputs(rows, rows)
    if rows == 200:
        logits[:, offset] += 30.0  # one expert on every row: more than one tile
    ids, _ = noaux_topk(logits, bias, 1.0)
    tile_expert, tile_limit, e2p, p2e, total, num_tiles = token_ordered_sort(ids, offset, local)
    d_expert, d_limit, d_e2p, d_p2e = _dict_sort(ids.tolist(), offset, local)
    n = len(d_expert)
    assert int(num_tiles) == n and int(total) == n * 128
    assert tile_expert[:n].tolist() == d_expert and tile_limit[:n].tolist() == d_limit
    assert (tile_expert[n:] == -1).all() and (tile_limit[n:] == -1).all()
    flat = e2p.reshape(-1).tolist()
    assert all(flat[x] == d_e2p.get(x, -1) for x in range(len(flat)))
    assert {i: v for i, v in enumerate(p2e.tolist()) if v >= 0} == d_p2e
    max_tiles, max_permuted = sort_buffer_sizes(rows, local)
    assert tile_expert.numel() == max_tiles and p2e.numel() == max_permuted


@pytest.mark.cpu_only
def test_reference_takes_split_k_partials_in_split_order():
    logits, bias = _inputs(4, 9)
    parts = torch.stack((logits * 0.25, logits * 0.75, torch.zeros_like(logits)))
    whole = rubin_route_tail.reference(parts[0] + parts[1] + parts[2], bias, 1.0, 0, 112)
    split = rubin_route_tail.reference(parts, bias, 1.0, 0, 112)
    assert all(torch.equal(a, b) for a, b in zip(whole, split))
    assert rubin_route_tail.reference(logits, bias, 1.0, 0, 112, num_tokens=2)[0].shape == (
        2,
        TOP_K,
    )
