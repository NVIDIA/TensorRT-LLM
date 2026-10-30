# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the swiglu_fp32 entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.swiglu_fp32 import (  # noqa: E501
    swiglu_fp32 as op,
)

__extra_import_path__ = ["..", "../../../.."]
import _expert_cases as cases  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "swiglu_fp32 requires a CUDA device"


@pytest.fixture(scope="module", autouse=True)
def controls():
    with cases.numerical_controls():
        yield


def _build(spec, seed):
    """Gate and up as the expert's projections produce them: unit-variance bf16."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (spec["tokens"], spec["intermediate"])
    gate = torch.randn(shape, generator=g).to(spec["dtype"]).cuda()
    up = torch.randn(shape, generator=g).to(spec["dtype"]).cuda()
    return gate, up


@pytest.mark.parametrize("seed", cases.SEEDS)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:26] for c in op.CELLS])
def test_certified_cells(cell, seed, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """SiLU and product within 2^-16 relative of the fp64 steps (fp32 formula spread on the
    same inputs), the activation within one bf16 unit (a product that close to a rounding
    boundary may round either way)."""
    certifying(op)
    gate, up = _build(cell.spec, seed)
    out = op(gate, up)
    ref = op.reference(gate, up)
    assert {k: v.dtype for k, v in out.items()} == {
        "silu": torch.float32,
        "product": torch.float32,
        "act": torch.bfloat16,
    }
    for name, rtol in (("silu", cases.FP32_FORMULA), ("product", cases.FP32_FORMULA)):
        torch.testing.assert_close(out[name], ref[name], rtol=rtol, atol=0.0)
    torch.testing.assert_close(out["act"], ref["act"], rtol=cases.BF16_UNIT, atol=0.0)


def test_bf16_steps_are_a_different_answer() -> None:
    """On the precision fixture's pairs every activation is exact here, and every one
    differs from the activation of a bf16 SiLU times up computed in bf16."""
    _, x, expected, premature_act = cases.precision_fixture()
    gate, up = x[:, : cases.INTER].cuda(), x[:, cases.INTER :].cuda()
    out = op(gate, up)
    assert torch.equal(out["act"].cpu(), expected["act"])
    bf16_steps = torch.nn.functional.silu(gate) * up
    assert torch.equal(bf16_steps.cpu(), premature_act)
    assert bool((out["act"] != bf16_steps).all())


@pytest.mark.parametrize(
    "case,match",
    [
        ("broadcast_up", "must have one shape"),
        ("fp32_gate", "must be torch.bfloat16"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    gate, up = _build(op.CELLS[1].spec, seed=17)
    if case == "broadcast_up":
        up = up[:1]
    else:
        gate = gate.float()
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(gate, up)
