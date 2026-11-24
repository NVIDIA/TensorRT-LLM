# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 route_topk entry: every cell against the plain-torch reference."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.route_topk import (  # noqa: E501
    k31_route_topk as op,
)
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

if not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("route_topk runs a CuTe DSL kernel", allow_module_level=True)
assert torch.cuda.is_available(), "route_topk requires a CUDA device"

# Rows per rank of the attention-DP MoE (decode, and DFlash2 verify at 1 + 7 rows per
# sequence), at the Kimi K3.1 checkpoint's scale of 1.0.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31/kimi_k31_pro_mxfp4__sm_10x__default": (
        Cell(
            why="DEP decode, 96 sequences on a rank: 96 rows",
            spec=dict(rows=96, scale=1.0),
        ),
        Cell(
            why="DEP DFlash2 verify, 14 sequences x 8 rows: 112 rows",
            spec=dict(rows=112, scale=1.0),
        ),
        Cell(
            why="DEP DFlash2 verify, 128 sequences x 8 rows: 1024 rows",
            spec=dict(rows=1024, scale=1.0),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)

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


def test_bias_steers_selection_not_weights() -> None:
    """A large bias on one expert selects it everywhere; its weight is still its sigmoid share."""
    logits, bias, _ = _build(dict(rows=80, scale=1.0), seed=5)
    bias.zero_()
    bias[123] = 100.0
    ids, weights = op(logits, bias, 1.0)
    ref_ids, ref_weights = op.reference(logits, bias, 1.0)
    assert (ids[:, 0] == 123).all()
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights.view(torch.int16), ref_weights.view(torch.int16))
