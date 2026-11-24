# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_route_topk_wide entry: every cell, and a context rank's rows under every score shape,
return noaux_tc_op's experts and FP32 weights bit for bit on SM107, whatever the grid's shape;
the guard and the row bound on the CPU."""

import re
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    rubin_route_topk_wide as _module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import (
    NUM_EXPERTS,
    TOP_K,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_route_topk_wide import (  # noqa: E501
    MAX_M,
    rubin_route_topk_wide,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"
)
gpu = pytest.mark.skipif(not _SM107, reason="needs an SM107 GPU")

#: A DEP16 / DEP12 / DEP8 context rank's chunk (max_num_tokens).
CTX_ROWS = 16384


def _inputs(rows, seed, mode="realistic", device="cpu"):
    """``(logits, bias)``, FP32. ``realistic``: router-like logits (requests of 8 rows share most
    of their hidden state, hot experts) and a bias spread of ~0.19, as K3.1's; ``rand``; ``skew``
    (eight hot experts); ``saturated`` (logits of +-40, where the tanh-form sigmoid is exactly 0
    or 1); ``ties`` (rows of few distinct values: the exact arg-max fallback, ties to the lower
    expert)."""
    g = torch.Generator(device=device).manual_seed(seed)
    if mode == "ties":
        logits = torch.zeros(rows, NUM_EXPERTS, device=device)
        half = rows // 2
        logits[:half] = torch.randint(
            -2, 3, (half, NUM_EXPERTS), generator=g, device=device
        ).float()
        logits[half:, ::3] = 1.0
        return logits, torch.zeros(NUM_EXPERTS, device=device)
    if mode == "realistic":
        dim = 512
        directions = torch.randn(NUM_EXPERTS, dim, generator=g, device=device)
        directions /= directions.norm(dim=1, keepdim=True)
        popularity = torch.randn(NUM_EXPERTS, generator=g, device=device) * 0.7
        hidden = torch.randn((rows + 7) // 8, dim, generator=g, device=device)
        hidden = hidden.repeat_interleave(8, 0)[:rows] * 0.85
        hidden = hidden + torch.randn(rows, dim, generator=g, device=device) * 0.5
        hidden /= hidden.norm(dim=1, keepdim=True)
        logits = (hidden @ directions.t()) * 22.0 + popularity - 1.5
        bias = (torch.rand(NUM_EXPERTS, generator=g, device=device) - 0.5) * 0.19
        return logits.contiguous(), bias
    logits = torch.randn(rows, NUM_EXPERTS, generator=g, device=device)
    bias = torch.randn(NUM_EXPERTS, generator=g, device=device) * 0.01
    if mode == "rand":
        logits *= 2.0
    elif mode == "skew":
        logits[:, 224:232] += 6.0
    elif mode == "saturated":
        logits *= 40.0
    else:
        raise ValueError(mode)
    return logits, bias


def _noaux(logits, bias, scale):
    """The path the entry replaces on a context server: ``noaux_tc_op`` with one expert group."""
    import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers noaux_tc_op

    weights, ids = torch.ops.trtllm.noaux_tc_op(logits, bias, 1, 1, TOP_K, scale)
    return ids, weights


def _assert_bitwise(got, want):
    ids, weights = got
    want_ids, want_weights = want
    assert ids.dtype == want_ids.dtype == torch.int32, (ids.dtype, want_ids.dtype)
    assert weights.dtype == want_weights.dtype == torch.float32, (weights.dtype, want_weights.dtype)
    assert ids.shape == want_ids.shape == weights.shape == want_weights.shape
    assert torch.equal(ids, want_ids), "experts (or their order) differ from noaux_tc_op"
    assert torch.equal(weights.view(torch.int32), want_weights.view(torch.int32)), (
        "FP32 weights differ from noaux_tc_op's"
    )


@gpu
@pytest.mark.parametrize(
    "cell", rubin_route_topk_wide.CELLS, ids=[c.why[:48] for c in rubin_route_topk_wide.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_route_topk_wide)
    spec = cell.spec
    m, scale = spec["rows"], spec.get("scale", 1.0)
    seed = zlib.crc32(cell.why.encode())
    logits, bias = _inputs(m, seed, spec.get("mode", "realistic"), "cuda")
    got = rubin_route_topk_wide(logits, bias, scale)
    torch.cuda.synchronize()
    _assert_bitwise(got, _noaux(logits, bias, scale))
    ref_ids, ref_weights = rubin_route_topk_wide.reference(logits, bias, scale)
    assert torch.equal(got[0], ref_ids), "selected experts differ from the reference"
    # The reference's FP32 sum of the 16 picks may take another order: two ulp at most.
    torch.testing.assert_close(got[1], ref_weights, rtol=2**-22, atol=0)


@gpu
@pytest.mark.parametrize("mode", ["realistic", "rand", "skew", "saturated", "ties"])
@pytest.mark.parametrize("rows", [CTX_ROWS, 12000, 257])
def test_context_rows_bitwise_to_noaux(rows, mode):
    """A context rank's rows under every score shape: noaux_tc_op's bits."""
    logits, bias = _inputs(rows, 31 * rows + len(mode), mode, "cuda")
    _assert_bitwise(rubin_route_topk_wide(logits, bias, 1.0), _noaux(logits, bias, 1.0))


@gpu
@pytest.mark.parametrize(
    "tokens_per_cta,num_warps,warps_per_token",
    [(8, 8, 1), (16, 16, 1), (16, 8, 1), (32, 8, 1), (32, 4, 1), (16, 16, 2), (16, 8, 4)],
)
def test_every_grid_shape_gives_the_same_bits(tokens_per_cta, num_warps, warps_per_token):
    """The grid's shape changes which warps route a row, never the row's result."""
    logits, bias = _inputs(CTX_ROWS, 7, "realistic", "cuda")
    saved = _module.LAUNCH
    _module.LAUNCH = (tokens_per_cta, num_warps, warps_per_token)
    try:
        got = rubin_route_topk_wide(logits, bias, 1.0)
    finally:
        _module.LAUNCH = saved
    _assert_bitwise(got, _noaux(logits, bias, 1.0))


@gpu
def test_row_slices_of_a_larger_buffer():
    """Leading, trailing and inner 16-byte-aligned row slices route as their own rows."""
    logits, bias = _inputs(10000, 21, "realistic", "cuda")
    for lo, hi in ((0, 5000), (5000, 10000), (1, 7)):
        part = logits[lo:hi]
        _assert_bitwise(
            rubin_route_topk_wide(part, bias, 1.0), _noaux(part.contiguous(), bias, 1.0)
        )


@gpu
def test_cuda_graph_replay_on_new_logits():
    """No state between launches: a captured launch replays on new logits."""
    logits, bias = _inputs(4096, 11, device="cuda")
    rubin_route_topk_wide(logits, bias, 1.0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, weights = rubin_route_topk_wide(logits, bias, 1.0)
    for seed in (12, 13):
        logits.copy_(_inputs(4096, seed, device="cuda")[0])
        graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise((ids, weights), _noaux(logits, bias, 1.0))


@gpu
@pytest.mark.parametrize("rows", [0, MAX_M + 1])
def test_the_launcher_refuses_rows_outside_its_bound(rows):
    logits = torch.zeros(rows, NUM_EXPERTS, device="cuda")
    with pytest.raises(ValueError):
        rubin_route_topk_wide(logits, torch.zeros(NUM_EXPERTS, device="cuda"), 1.0)


@pytest.mark.cpu_only
def test_the_row_bound_is_the_kernels():
    """The entry's bound is the kernel's top-k-only bound (read from the kernel's source)."""
    kernel = Path(_module.__file__).resolve().parents[2] / "kernels/moe/rubin_route_tail.py"
    bound = re.search(r"^WIDE_TOPK_MAX_M = (\d+)$", kernel.read_text(), re.MULTILINE)
    assert bound is not None and int(bound.group(1)) == MAX_M


@pytest.mark.cpu_only
@pytest.mark.parametrize("rows", [1, 2049, CTX_ROWS])
def test_guard_admits_a_context_ranks_call(rows):
    logits = torch.zeros(rows, NUM_EXPERTS)
    rubin_route_topk_wide.is_valid(logits, torch.zeros(NUM_EXPERTS), 1.0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case",
    [
        "no_rows",
        "too_many_rows",
        "three_d",
        "width",
        "bf16_logits",
        "strided_logits",
        "bias_size",
        "bf16_bias",
    ],  # fmt: skip
)
def test_guard_refuses_what_the_launcher_misreads(case):
    logits, bias = torch.zeros(96, NUM_EXPERTS), torch.zeros(NUM_EXPERTS)
    if case == "no_rows":
        logits = logits[:0]
    elif case == "too_many_rows":
        logits = torch.zeros(1, NUM_EXPERTS).expand(MAX_M + 1, NUM_EXPERTS)
    elif case == "three_d":
        logits = torch.zeros(2, 96, NUM_EXPERTS)
    elif case == "width":
        logits = torch.zeros(96, 512)
    elif case == "bf16_logits":
        logits = logits.to(torch.bfloat16)
    elif case == "strided_logits":
        logits = torch.zeros(96, 2 * NUM_EXPERTS)[:, :NUM_EXPERTS]
    elif case == "bias_size":
        bias = bias[:448]
    elif case == "bf16_bias":
        bias = bias.to(torch.bfloat16)
    with pytest.raises(AssertionError):
        rubin_route_topk_wide.is_valid(logits, bias, 1.0)
