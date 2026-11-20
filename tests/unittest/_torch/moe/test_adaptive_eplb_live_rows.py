# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Under a live-row word (R real of P requests in bits 0-15 / 16-31;
tensorrt_llm/_torch/moe/live_rows.py) the adaptive EPLB route stages the rows of padding requests as
trtllm::megamoe_stage_routes does, and its statistics count only the live rows."""

import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference
from _torch.moe.test_adaptive_eplb_kernels import TOP_K, _dsv4_case, _stats, _tables, needs_gpu

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import build_route_tables


def _stage_routes(ids, weights, bucket, word):
    """What the MegaMoE staging holds after trtllm::megamoe_stage_routes of these routes."""
    idx = torch.full((bucket, TOP_K), 7, dtype=ids.dtype, device=ids.device)
    out = torch.full((bucket, TOP_K), 5.0, device=ids.device)
    torch.ops.trtllm.megamoe_stage_routes(ids, weights, idx, out, word)
    return idx, out


def _dropped(routes, count):
    """Dispatch and top-k reduce drop slots outside the unsigned range [0, count)."""
    return (routes < 0) | (routes >= count)


@needs_gpu
@pytest.mark.parametrize(
    "real,requests",
    # 32 rows: R of 8 requests keep 4 R rows; no request or 7 requests (not dividing 32) keep 0 or 32.
    [(8, 8), (5, 8), (0, 8), (0, 0), (3, 7)],
)
@pytest.mark.parametrize("fused", [False, True])
def test_the_route_drops_the_padding_rows_as_the_megamoe_staging_does(real, requests, fused):
    """Regression: record counts only live rows. Separate and fused routes preserve live-word
    staging and statistics bit for bit."""
    layout, shares, ids = _dsv4_case(2)
    tables = _tables(*build_route_tables(layout, shares, 384, 512))
    ids = torch.tensor(ids[: 32 * TOP_K], dtype=torch.int32, device="cuda").view(32, TOP_K)
    weights = torch.rand(32, TOP_K, device="cuda")
    word = torch.tensor([(requests << 16) | real], dtype=torch.int32, device="cuda")
    slots = torch.full((48, TOP_K), 9, dtype=torch.int32, device="cuda")
    slot_weights = torch.full((48, TOP_K), 3.0, device="cuda")
    routed, recorded, expected = _stats(384), _stats(384), _stats(384)
    x = torch.zeros(32, 7168, dtype=torch.bfloat16, device="cuda")
    quantize = (
        (
            x,
            torch.ones(1, device="cuda"),
            x.new_empty(32, 3584, dtype=torch.uint8),
            x.new_empty(32, 448, dtype=torch.uint8),
        )
        if fused
        else ()
    )
    torch.ops.trtllm.adaptive_eplb_route_into(
        ids, *tables, 3, slots, weights, slot_weights, *routed, True, word, *quantize
    )
    plain = torch.empty_like(ids)
    torch.ops.trtllm.adaptive_eplb_route_into(ids, *tables, 3, plain)
    on_slots, on_weights = _stage_routes(plain, weights, 48, word)
    off_experts, off_weights = _stage_routes(ids, weights, 48, word)
    bits = slot_weights.view(torch.int32)
    assert torch.equal(slots, on_slots) and torch.equal(bits, on_weights.view(torch.int32))
    assert torch.equal(_dropped(slots, 512), _dropped(off_experts, 384))
    assert torch.equal(bits, off_weights.view(torch.int32))
    torch.ops.trtllm.adaptive_eplb_record(ids, *recorded, True, word)
    reference.record(ids.cpu(), *expected, True, word.cpu())
    for stats in (routed, recorded):
        assert torch.equal(stats.rows.view(torch.int32), expected.rows.view(torch.int32))
        assert torch.equal(stats.forwards, expected.forwards)
