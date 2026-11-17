# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sm_107 tp4 target's decode MoE routing: the runner routing the fp32 router logits itself
against `renorm_moe_routing_op` ahead of the pre-routed runner, on one GPU.

The decode target hands the NVFP4 runner the router's fp32 logits and the routing method
`_MOE_ROUTING_METHOD` (Renormalize); the prefill target routes with `renorm_moe_routing_op` and
hands the runner ids and weights. Both must select the same experts in the same slot order
with the same weights. With random NVFP4 experts no two experts give the same row, so two
outputs are compared bit for bit:

- the finalized rows (`do_finalize=True`): the routed partial the layer adds;
- per (token, slot), the unweighted row of the expert that slot gathered (`do_finalize=False`,
  read through its expanded-to-permuted index), which pins the expert id and the slot order.

The logits cover the cases where routings part ways: well-spread logits, the 4th and 5th of
every row one fp32 ulp apart, logits of magnitude 100, a few integer levels with many exact
ties, and all zeros. Expert slices are the target's at four ranks (1536 wide) and eight (768).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import (
    fp4_block_scale_moe_runner,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.mistral_large4.mistral_large_4_nvfp4__sm_107__tp4.modeling import (  # noqa: E501
    _MOE_ACT_SWIGLU,
    _MOE_ROUTING_METHOD,
    _NVFP4_SF_LINEAR,
    _NVFP4_SF_VEC,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 tp4 target's decode routing requires SM107",
)

HIDDEN = 6144
EXPERTS = 128
TOPK = 4
ROWS = (1, 2, 3, 4, 8, 16, 32, 64, 128, 1024)
DISTRIBUTIONS = ("spread", "near_tie", "wide", "coarse", "zeros")


def _e4m3(shape, gen) -> torch.Tensor:
    return (torch.rand(shape, generator=gen, device="cuda") * 0.75 + 0.25).to(torch.float8_e4m3fn)


@pytest.fixture(scope="module", params=(1536, 768), ids=("tp4", "tp8"))
def experts(request):
    """All 128 experts at one rank's slice width, random NVFP4 codes and scales."""
    inter = request.param
    gen = torch.Generator(device="cuda").manual_seed(4242 + inter)
    return dict(
        inter=inter,
        fc1_w=torch.randint(
            0,
            256,
            (EXPERTS, 2 * inter, HIDDEN // 2),
            generator=gen,
            device="cuda",
            dtype=torch.uint8,
        ),
        fc1_s=_e4m3((EXPERTS, 2 * inter, HIDDEN // _NVFP4_SF_VEC), gen),
        fc2_w=torch.randint(
            0, 256, (EXPERTS, HIDDEN, inter // 2), generator=gen, device="cuda", dtype=torch.uint8
        ),
        fc2_s=_e4m3((EXPERTS, HIDDEN, inter // _NVFP4_SF_VEC), gen),
        out1=torch.full((EXPERTS,), 2e-3, dtype=torch.float32, device="cuda"),
        out1_gate=torch.full((EXPERTS,), 2e-3, dtype=torch.float32, device="cuda"),
        out2=torch.full((EXPERTS,), 1e-3, dtype=torch.float32, device="cuda"),
    )


def _logits(dist: str, rows: int, gen: torch.Generator) -> torch.Tensor:
    if dist == "spread":
        return torch.randn(rows, EXPERTS, generator=gen, device="cuda") * 3
    if dist == "wide":
        return torch.randn(rows, EXPERTS, generator=gen, device="cuda") * 100
    if dist == "coarse":
        return torch.randint(0, 3, (rows, EXPERTS), generator=gen, device="cuda").float()
    if dist == "zeros":
        return torch.zeros(rows, EXPERTS, device="cuda")
    assert dist == "near_tie"
    logits = torch.randn(rows, EXPERTS, generator=gen, device="cuda")
    top = logits.topk(TOPK + 1, dim=-1)
    fourth = top.values[:, TOPK - 1]
    logits[torch.arange(rows, device="cuda"), top.indices[:, TOPK]] = torch.nextafter(
        fourth, torch.full_like(fourth, -torch.inf)
    )
    return logits


def _run(e, logits, xq, xs, runner_routes: bool, finalize: bool):
    """The runner as the decode target calls it (`runner_routes`) or as the prefill target does."""
    ids = weights = None
    if not runner_routes:
        ids, weights = renorm_moe_routing_op(logits, TOPK, torch.bfloat16)
        logits = None
    return fp4_block_scale_moe_runner(
        logits, None, xq, xs, e["fc1_w"], e["fc1_s"], None, None, None, None, e["fc2_w"],
        e["fc2_s"], None, e["out1"], e["out1_gate"], e["out2"], EXPERTS, TOPK, None, None,
        e["inter"], 0, EXPERTS, None, _MOE_ROUTING_METHOD, finalize, _MOE_ACT_SWIGLU,
        topk_weights=weights, topk_ids=ids, tune_max_num_tokens=max(ROWS),
    )  # fmt: skip


def _slot_rows(e, logits, xq, xs, runner_routes: bool, rows: int) -> torch.Tensor:
    """Per (token, slot), the unweighted expert row that slot gathered."""
    permuted, _, expanded_to_permuted = _run(e, logits, xq, xs, runner_routes, finalize=False)
    return permuted[expanded_to_permuted.reshape(-1).long()].reshape(rows, TOPK, HIDDEN)


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_runner_routing_matches_the_separate_routing(experts, dist, rows) -> None:
    gen = torch.Generator(device="cuda").manual_seed(rows * 31 + DISTRIBUTIONS.index(dist))
    logits = _logits(dist, rows, gen).contiguous()
    x = torch.randn(rows, HIDDEN, generator=gen, device="cuda").to(torch.bfloat16)
    gscale = torch.tensor([134.4], dtype=torch.float32, device="cuda")
    xq, xs = fp4_quantize(x, gscale, _NVFP4_SF_VEC, False, _NVFP4_SF_LINEAR)
    xs = xs.view(torch.float8_e4m3fn)

    want = _run(experts, logits, xq, xs, runner_routes=False, finalize=True)[0]
    got = _run(experts, logits, xq, xs, runner_routes=True, finalize=True)[0]
    differ = (got.view(torch.int16) != want.view(torch.int16)).any(-1)
    assert not differ.any(), f"{int(differ.sum())} of {rows} finalized rows differ"

    want = _slot_rows(experts, logits, xq, xs, False, rows)
    got = _slot_rows(experts, logits, xq, xs, True, rows)
    differ = (got.view(torch.int16) != want.view(torch.int16)).any(-1)
    assert not differ.any(), (
        f"{int(differ.sum())} of {rows * TOPK} (token, slot) pairs gathered another expert row"
    )
