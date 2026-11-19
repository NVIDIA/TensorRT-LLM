# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 route_topk entry: a sample of its cells against the plain-torch reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.route_topk import (  # noqa: E501
    k31_route_topk as op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("route_topk runs a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "route_topk requires a CUDA device"

CELLS = sample_cells(op.CELLS, size_key="rows")[:2]

#: Routed experts the kernel is built for (Kimi K3.1).
NUM_EXPERTS = 896


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(spec["rows"], NUM_EXPERTS, generator=g, device="cuda", dtype=torch.float32)
    bias = torch.randn(NUM_EXPERTS, generator=g, device="cuda", dtype=torch.float32) * 0.1
    return logits, bias, spec["scale"]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        ids, weights = op(*args)
    ref_ids, ref_weights = op.reference(*args)
    # Both claims are exact: the same experts in the same order, and the BF16 rounding of the
    # same FP32 weights.
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert ids.dtype is torch.int32 and weights.dtype is torch.bfloat16
    assert torch.equal(weights.view(torch.int16), ref_weights.view(torch.int16)), (
        "BF16 weights differ from the reference's rounding"
    )
