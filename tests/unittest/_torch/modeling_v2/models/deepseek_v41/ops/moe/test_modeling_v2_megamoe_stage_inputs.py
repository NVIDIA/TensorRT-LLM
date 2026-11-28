# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the megamoe_stage_inputs entry."""

import pytest
import torch

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_stage_inputs import (  # noqa: E402
    megamoe_stage_inputs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = "cuda"
MAX_T, TOP_K, HIDDEN = 64, 6, 5120


def _buffers():
    idx = torch.full((MAX_T, TOP_K), 12345, dtype=torch.int64, device=DEV)
    w = torch.full((MAX_T, TOP_K), 9.0, dtype=torch.float32, device=DEV)
    act = torch.full((MAX_T, HIDDEN // 2), 0x77, dtype=torch.uint8, device=DEV)
    sf = torch.full((MAX_T, HIDDEN // 16), 0x66, dtype=torch.uint8, device=DEV)
    return idx, w, act, sf


@pytest.mark.parametrize(
    "cell", megamoe_stage_inputs.CELLS, ids=[c.why[:44] for c in megamoe_stage_inputs.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(megamoe_stage_inputs)
    n, max_t = cell.spec["rows"], cell.spec["max_t"]
    g = torch.Generator(device=DEV).manual_seed(n + max_t)
    slots = torch.randint(0, 400, (n, TOP_K), device=DEV, generator=g, dtype=torch.int32)
    weights = torch.rand(n, TOP_K, device=DEV, generator=g)
    x = torch.randint(0, 256, (n, HIDDEN // 2), device=DEV, generator=g, dtype=torch.uint8)
    xsf = torch.randint(0, 256, (n, HIDDEN // 16), device=DEV, generator=g, dtype=torch.uint8)
    idx = torch.full((max_t, TOP_K), 12345, dtype=torch.int64, device=DEV)
    w = torch.full((max_t, TOP_K), 9.0, dtype=torch.float32, device=DEV)
    act = torch.full((max_t, HIDDEN // 2), 0x77, dtype=torch.uint8, device=DEV)
    sf = torch.full((max_t, HIDDEN // 16), 0x66, dtype=torch.uint8, device=DEV)
    megamoe_stage_inputs(slots, weights, idx, w, x, xsf, act, sf)
    ref_idx, ref_w = megamoe_stage_inputs.reference(slots, weights, max_t)
    assert torch.equal(idx, ref_idx) and torch.equal(w, ref_w)
    assert torch.equal(act[:n], x) and torch.equal(sf[:n], xsf)


@pytest.mark.parametrize("n", [0, 1, 17, MAX_T])
def test_live_rows_are_copied_and_the_rest_are_idle(n):
    g = torch.Generator(device=DEV).manual_seed(n)
    slots = torch.randint(0, 400, (n, TOP_K), device=DEV, generator=g, dtype=torch.int32)
    weights = torch.rand(n, TOP_K, device=DEV, generator=g)
    x = torch.randint(0, 256, (n, HIDDEN // 2), device=DEV, generator=g, dtype=torch.uint8)
    xsf = torch.randint(0, 256, (n, HIDDEN // 16), device=DEV, generator=g, dtype=torch.uint8)
    idx, w, act, sf = _buffers()
    megamoe_stage_inputs(slots, weights, idx, w, x, xsf, act, sf)
    assert torch.equal(idx[:n], slots.long()) and (idx[n:] == -1).all()
    assert torch.equal(w[:n], weights) and (w[n:] == 0).all()
    assert torch.equal(act[:n], x) and torch.equal(sf[:n], xsf)
    assert (act[n:] == 0x77).all() and (sf[n:] == 0x66).all(), (
        "activation rows past n are not written"
    )


def test_routing_only_form():
    slots = torch.randint(0, 400, (5, TOP_K), device=DEV, dtype=torch.int32)
    weights = torch.rand(5, TOP_K, device=DEV)
    idx, w, act, sf = _buffers()
    megamoe_stage_inputs(slots, weights, idx, w)
    assert torch.equal(idx[:5], slots.long()) and (idx[5:] == -1).all() and (act == 0x77).all()
    assert torch.equal(w[:5], weights) and (w[5:] == 0).all(), (
        "the routing-only form stages the weights too"
    )


def test_graph_capturable_and_unaligned_inputs_are_refused():
    slots = torch.randint(0, 400, (9, TOP_K), device=DEV, dtype=torch.int32)
    weights = torch.rand(9, TOP_K, device=DEV)
    x = torch.randint(0, 256, (9, HIDDEN // 2), device=DEV, dtype=torch.uint8)
    xsf = torch.randint(0, 256, (9, HIDDEN // 16), device=DEV, dtype=torch.uint8)
    idx, w, act, sf = _buffers()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        megamoe_stage_inputs(slots, weights, idx, w, x, xsf, act, sf)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        megamoe_stage_inputs(slots, weights, idx, w, x, xsf, act, sf)
    slots.copy_(torch.randint(0, 400, (9, TOP_K), device=DEV, dtype=torch.int32))
    graph.replay()
    assert torch.equal(idx[:9], slots.long())
    with pytest.raises(AssertionError, match="contiguous"):
        megamoe_stage_inputs.is_valid(slots.t().contiguous().t(), weights, idx, w)
    with pytest.raises(AssertionError, match="go together"):
        megamoe_stage_inputs.is_valid(
            slots, weights, idx, w, x, xsf
        )  # activation without its buffers
