# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_route_topk_out entry: every cell writes noaux_tc_op's bits into the caller's rows on
SM107, and nothing past them (of split-K partials: noaux_tc_op's bits on their sum in split
order); the guard on the CPU."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (
    NUM_EXPERTS,
    TOP_K,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_route_topk_out import (  # noqa: E501
    rubin_route_topk_out,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

STALE_ID = -7


def _inputs(rows, seed, mode="realistic", device="cpu"):
    """Router-like logits (requests of 8 rows share most of their hidden state, hot experts) and a
    bias spread of ~0.19, as K3.1's; or tied scores (the rubin_route_tail entry test's inputs)."""
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


def _partials(logits, splits, partial_rows, seed):
    """``splits`` split-K partials ``[S, R, 896]`` (R = ``partial_rows``) whose first rows sum, in
    split order, to about ``logits``; and that sum over ``logits``' rows, FP32."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    m, device = logits.shape[0], logits.device
    full = torch.zeros(partial_rows, NUM_EXPERTS, device=device)
    full[:m] = logits
    noise = (torch.randn(splits - 1, partial_rows, NUM_EXPERTS, generator=g) * 0.5).to(device)
    parts = torch.cat([(full - noise.sum(0))[None], noise]).contiguous()
    summed = parts[0, :m].clone()
    for part in parts[1:]:
        summed = summed + part[:m]
    return parts, summed


def _outputs(out_rows, device):
    return (
        torch.full((out_rows, TOP_K), STALE_ID, dtype=torch.int32, device=device),
        torch.full((out_rows, TOP_K), float("nan"), device=device),
    )


def _check_today(m, logits, bias, scale, ids, weights):
    """noaux_tc_op's bits in rows [0, m), the stale values past them."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers noaux_tc_op

    want_w, want_ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, scale)
    assert torch.equal(ids[:m], want_ids.to(torch.int32)), "experts differ from noaux_tc_op"
    assert torch.equal(weights[:m].view(torch.int32), want_w.float().view(torch.int32)), (
        "weights differ from noaux_tc_op"
    )
    assert bool((ids[m:] == STALE_ID).all()) and bool(weights[m:].isnan().all()), (
        "rows past the logits were written"
    )


@gpu
@pytest.mark.parametrize(
    "cell", rubin_route_topk_out.CELLS, ids=[c.why[:48] for c in rubin_route_topk_out.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_route_topk_out)
    spec = cell.spec
    m, scale = spec["rows"], spec.get("scale", 1.0)
    logits, bias = _inputs(m, zlib.crc32(cell.why.encode()), spec.get("mode", "realistic"), "cuda")
    ids, weights = _outputs(spec["out_rows"], "cuda")
    approx, tail = spec.get("route_approx", False), spec.get("tail")
    kw = dict(route_approx=approx, tail=tail)
    if spec.get("splits", 1) > 1:
        logits, summed = _partials(logits, spec["splits"], spec.get("partial_rows", m), m)
        kw["rows"] = m
    else:
        summed = logits
    rubin_route_topk_out(logits, bias, scale, ids, weights, **kw)
    torch.cuda.synchronize()
    _check_today(m, summed.contiguous(), bias, scale, ids, weights)
    ref_ids, ref_weights = rubin_route_topk_out.reference(logits, bias, scale, ids, weights, **kw)
    assert torch.equal(ids[:m], ref_ids), "selected experts differ from the reference"
    # The reference's FP32 sum of the 16 picks may take another order: two ulp at most.
    torch.testing.assert_close(weights[:m], ref_weights, rtol=2**-22, atol=0)


@gpu
@pytest.mark.parametrize("rows,out_rows", [(96, 128), (512, 1024)])
def test_cuda_graph_replay_on_new_logits(rows, out_rows):
    """Captured after one eager call; each replay on new logits writes their routing."""
    logits, bias = _inputs(rows, 11, device="cuda")
    ids, weights = _outputs(out_rows, "cuda")
    rubin_route_topk_out(logits, bias, 1.0, ids, weights, route_approx=rows <= 256)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rubin_route_topk_out(logits, bias, 1.0, ids, weights, route_approx=rows <= 256)
    for seed in (12, 13):
        logits.copy_(_inputs(rows, seed, device="cuda")[0])
        graph.replay()
        torch.cuda.synchronize()
        _check_today(rows, logits, bias, 1.0, ids, weights)


@pytest.mark.cpu_only
def test_guard_admits_the_dep16_call():
    logits, bias = _inputs(96, 1)
    ids, weights = _outputs(128, "cpu")
    rubin_route_topk_out.is_valid(logits, bias, 1.0, ids[:96], weights[:96], route_approx=True)
    rubin_route_topk_out.is_valid(logits, bias, 1.0, ids, weights)


@pytest.mark.cpu_only
def test_guard_admits_the_router_tf32_splitk_call():
    """dep16's routing of the router's 8 partials into rows [0, 96) of bucket 128."""
    logits, bias = _inputs(96, 1)
    parts, _ = _partials(logits, 8, 96, 1)
    ids, weights = _outputs(128, "cpu")
    rubin_route_topk_out.is_valid(
        parts, bias, 1.0, ids[:96], weights[:96], route_approx=True, rows=96, tail=(96, 4, 4)
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case",
    [
        "no_rows",
        "rows_past_partials",
        "rows_on_logits",
        "too_many_ctas",
        "team_size",
        "rows_per_cta",
    ],
)
def test_guard_refuses_a_partials_call_or_launch_the_launcher_misreads(case):
    logits, bias = _inputs(96, 3)
    parts, _ = _partials(logits, 8, 96, 3)
    ids, weights = _outputs(128, "cpu")
    kw = dict(rows=96, tail=(96, 4, 4))
    if case == "no_rows":
        kw["rows"] = 0
    elif case == "rows_past_partials":
        kw["rows"] = 97
    elif case == "rows_on_logits":
        parts = logits
    elif case == "too_many_ctas":
        kw["tail"] = (97, 4, 4)
    elif case == "team_size":
        kw["tail"] = (96, 6, 3)
    elif case == "rows_per_cta":
        kw["tail"] = (2, 4, 4)
    with pytest.raises(AssertionError):
        rubin_route_topk_out.is_valid(parts, bias, 1.0, ids, weights, **kw)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case",
    [
        "no_rows",
        "too_many_rows",
        "bf16_logits",
        "strided_logits",
        "short_ids",
        "ids_int64",
        "weights_bf16",
        "strided_weights",
        "bias_size",
    ],  # fmt: skip
)
def test_guard_refuses_what_the_launcher_misreads(case):
    logits, bias = _inputs(96, 2)
    ids, weights = _outputs(128, "cpu")
    if case == "no_rows":
        logits = logits[:0]
    elif case == "too_many_rows":
        logits = torch.zeros(2049, NUM_EXPERTS)
        ids, weights = _outputs(2049, "cpu")
    elif case == "bf16_logits":
        logits = logits.to(torch.bfloat16)
    elif case == "strided_logits":
        logits = torch.zeros(96, 2 * NUM_EXPERTS)[:, :NUM_EXPERTS]
    elif case == "short_ids":
        ids = ids[:64]
    elif case == "ids_int64":
        ids = ids.long()
    elif case == "weights_bf16":
        weights = weights.to(torch.bfloat16)
    elif case == "strided_weights":
        weights = torch.zeros(128, 2 * TOP_K)[:, :TOP_K]
    elif case == "bias_size":
        bias = bias[:448]
    with pytest.raises(AssertionError):
        rubin_route_topk_out.is_valid(logits, bias, 1.0, ids, weights)
