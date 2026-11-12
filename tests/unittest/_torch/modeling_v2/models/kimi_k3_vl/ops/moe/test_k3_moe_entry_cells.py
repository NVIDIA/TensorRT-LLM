# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single-GPU MoE entries of the K3 default target over their certified cells:
``k3_router_logits.CELLS``, ``k3_route_tail.CELLS`` and ``k3_moe_residual_add.CELLS``, each driven
through ``validating`` and checked against the entry's own ``reference``.
"""

import importlib

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import ULP, assert_within_ulp, validating  # noqa: E402

K3 = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103, 107),
    reason="the K3 routing kernels run on SM100 / SM103 / SM107",
)

route_tail = importlib.import_module(f"{K3}.kernels.k3r_moe.k3_route_tail")
k3_router_logits = importlib.import_module(f"{K3}.ops.moe.k3_router_logits").k3_router_logits
k3_route_tail = importlib.import_module(f"{K3}.ops.moe.k3_route_tail").k3_route_tail
k3_moe_residual_add = importlib.import_module(
    f"{K3}.ops.moe.k3_moe_residual_add"
).k3_moe_residual_add

HIDDEN, NUM_EXPERTS, TOP_K = 7168, 896, 16
ROUTED_SCALING_FACTOR = 2.5


def _ids(cells):
    return [c.why[:48] for c in cells]


@pytest.mark.parametrize("cell", k3_router_logits.CELLS, ids=_ids(k3_router_logits.CELLS))
def test_k3_router_logits_cells(cell):
    """FP32 logits of BF16 operands: the products are exact in FP32 and only the summation order
    differs from the reference's (14 slices of 512, then the slices in order, against cuBLAS's
    FP32 order). A random-walk estimate of that difference over K = 7168 terms of these operands
    is ~1.6e-6 of the row's largest logit, about 13 FP32 ulps (2^-23); the gates allow 64 per
    element and 16 in RMS. The note's bitwise claim against ``dsv3_router_gemm_op`` is checked
    too."""
    m = cell.spec["m"]
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    w = (torch.randn(NUM_EXPERTS, HIDDEN, device="cuda", generator=gen) * 0.02).to(torch.bfloat16)
    with validating(k3_router_logits):
        got = k3_router_logits(x, w)
    want = k3_router_logits.reference(x, w)
    assert_within_ulp(got, want, 64, 16, ULP[torch.float32])
    stock = torch.ops.trtllm.dsv3_router_gemm_op(x, w.t(), bias=None, out_dtype=torch.float32)
    assert torch.equal(got, stock), "not bitwise dsv3_router_gemm_op"


@pytest.mark.parametrize("cell", k3_moe_residual_add.CELLS, ids=_ids(k3_moe_residual_add.CELLS))
def test_k3_moe_residual_add_cells(cell):
    """Two BF16 adds, each rounded once: bitwise the reference."""
    m = cell.spec["m"]
    gen = torch.Generator(device="cuda").manual_seed(m)
    routed, shared, residual = (
        torch.randn(m, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16) for _ in range(3)
    )
    with validating(k3_moe_residual_add):
        got = k3_moe_residual_add(routed, shared, residual)
    want = k3_moe_residual_add.reference(routed, shared, residual)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("cell", k3_route_tail.CELLS, ids=_ids(k3_route_tail.CELLS))
def test_k3_route_tail_cells(cell):
    """The top-k ids (selection and order) exactly; the weights within 4 FP32 ulps of the row's
    largest (both renormalize in FP64 from the same FP32 scores; the kernel's sigmoid is its own);
    of the ``moe_sort`` tables every written entry exactly (an expert's rows in ascending token
    order, so in ascending expanded index, as the reference sorts them)."""
    s = cell.spec
    m, offset, local, splits = s["m"], s["offset"], s["local"], s["splits"]
    gen = torch.Generator(device="cuda").manual_seed(1000 + m)
    shape = (splits, m, NUM_EXPERTS) if splits else (m, NUM_EXPERTS)
    logits = torch.randn(*shape, device="cuda", generator=gen)
    if splits:
        logits = logits / splits
    bias = 0.1 * torch.randn(NUM_EXPERTS, device="cuda", generator=gen)
    workspace = route_tail.wide_workspace(logits.device) if m > route_tail.MAX_M else None
    args = (logits, bias, ROUTED_SCALING_FACTOR, offset, local, route_tail.TILE)
    with validating(k3_route_tail):
        got = k3_route_tail(*args, workspace=workspace)
    torch.cuda.synchronize()
    want = k3_route_tail.reference(*args, workspace=workspace)
    ids, weights, tile_expert, tile_limit, e2p, p2e, total, num_tiles = got
    w_ids, w_weights, w_tile_expert, w_tile_limit, w_e2p, w_p2e, w_total, w_num_tiles = want

    assert torch.equal(ids, w_ids), "top-k ids"
    assert_within_ulp(weights, w_weights, 4, 1, ULP[torch.float32])
    tiles = int(w_num_tiles.item())
    assert int(num_tiles.item()) == tiles and int(total.item()) == int(w_total.item())
    assert torch.equal(tile_expert[:tiles], w_tile_expert[:tiles])
    assert torch.equal(tile_limit[:tiles], w_tile_limit[:tiles])
    local_rows = w_e2p.reshape(-1) >= 0
    assert bool(local_rows.any()), "the cell routes no row to the window"
    assert torch.equal(e2p.reshape(-1)[local_rows], w_e2p.reshape(-1)[local_rows])
    written = w_p2e >= 0
    assert torch.equal(p2e[: written.numel()][written], w_p2e[written])
