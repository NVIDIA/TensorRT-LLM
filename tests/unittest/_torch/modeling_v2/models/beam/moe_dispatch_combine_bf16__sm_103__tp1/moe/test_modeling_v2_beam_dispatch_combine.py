# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the beam_dispatch_combine entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_dispatch_combine import (  # noqa: E501
    beam_dispatch_combine_op as op,
)

__extra_import_path__ = ["..", "../../../.."]
import _dispatch_combine_cases as cases  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_dispatch_combine requires a CUDA device"


@pytest.mark.parametrize("seed", cases.SEEDS)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:26] for c in op.CELLS])
def test_certified_cells(cell, seed, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Bit for bit: both sides form the same correctly rounded fp32 products, round each
    route to bf16 once and add the routes in fp32 in the same expert order, so a single
    differing bit is a different computation, not a rounding allowance."""
    certifying(op)
    spec = cell.spec
    x, ids, w = (t.cuda() for t in cases.matrix_case(seed, spec["tokens"]))
    experts = cases.affine_experts()
    assert (spec["hidden_size"], spec["num_experts"], spec["top_k"]) == (
        cases.HIDDEN,
        cases.EXPERTS,
        cases.TOP_K,
    )
    called, referenced = experts.fresh(), experts.fresh()
    out = op(x, ids, w, called, spec["routed_scaling_factor"])
    ref = op.reference(x, ids, w, referenced, spec["routed_scaling_factor"])
    assert out.dtype == torch.bfloat16 and out.shape == (spec["tokens"], cases.HIDDEN)
    assert cases.bitwise_equal(out, ref)
    # One call per selected expert, ascending, on both sides.
    assert (
        [e for e, _, _ in called.calls]
        == [e for e, _, _ in referenced.calls]
        == sorted(set(ids.reshape(-1).tolist()))
    )


def test_reference_keeps_the_rounded_route() -> None:
    """On the cancellation witness the reference differs from the same combination with
    fp32 routes: rounding each weighted route to bf16 is the source's semantics."""
    x, ids, w, _ = (
        t.cuda() if isinstance(t, torch.Tensor) else t
        for t in cases.fixtures()["cancellation_witness"]
    )
    experts = cases.affine_experts()
    ref = op.reference(x, ids, w, experts.fresh(), cases.SCALE)
    gold = cases.golden(x, ids, w, experts.fresh())
    unrounded = cases.golden(x, ids, w, experts.fresh(), fault="no_route_cast")
    assert cases.bitwise_equal(ref, gold["output"])
    assert not torch.equal(ref, unrounded["output"])
