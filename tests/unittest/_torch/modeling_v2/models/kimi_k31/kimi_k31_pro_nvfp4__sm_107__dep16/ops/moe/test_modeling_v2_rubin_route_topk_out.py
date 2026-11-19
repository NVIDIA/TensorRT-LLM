# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rubin_route_topk_out entry: a sample of its cells writes noaux_tc_op's bits into the caller's rows on
SM107, and nothing past them."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__dep16.ops.moe.rubin_route_topk_out import (  # noqa: E501
    rubin_route_topk_out,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe._rubin_moe_common import TOP_K

__extra_import_path__ = ["../../../../..", "../../../kernels/moe", "../../.."]
from _k31_utils import cpu_or_sm, on_sm, sample_cells
from _rubin_route_tail_inputs import request_logits  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 -- path above; fixture by name

pytestmark = cpu_or_sm(107)
gpu = on_sm(107)

STALE_ID = -7


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


CELLS = sample_cells(rubin_route_topk_out.CELLS, size_key="rows")[:2]


@gpu
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 -- fixture resolved by name
    certifying(rubin_route_topk_out)
    spec = cell.spec
    m, scale = spec["rows"], spec.get("scale", 1.0)
    logits, bias = request_logits(
        m, zlib.crc32(cell.why.encode()), spec.get("mode", "realistic"), "cuda"
    )
    ids, weights = _outputs(spec["out_rows"], "cuda")
    rubin_route_topk_out(logits, bias, scale, ids, weights)
    torch.cuda.synchronize()
    _check_today(m, logits, bias, scale, ids, weights)
    ref_ids, ref_weights = rubin_route_topk_out.reference(logits, bias, scale, ids, weights)
    assert torch.equal(ids[:m], ref_ids), "selected experts differ from the reference"
    # The reference's FP32 sum of the 16 picks may take another order: two ulp at most.
    torch.testing.assert_close(weights[:m], ref_weights, rtol=2**-22, atol=0)
