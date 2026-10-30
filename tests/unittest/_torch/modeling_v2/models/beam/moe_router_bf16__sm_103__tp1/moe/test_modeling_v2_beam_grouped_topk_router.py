# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the beam_grouped_topk_router entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_grouped_topk_router import (  # noqa: E501
    beam_grouped_topk_router as op,
)

__extra_import_path__ = ["..", "../../../.."]
import _router_cases as cases  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "beam_grouped_topk_router requires a CUDA device"


def _build(spec, seed):
    weight, bias, hidden = cases.matrix_case(seed, spec["tokens"])
    kwargs = {k: spec[k] for k in ("n_group", "topk_group", "top_k", "eps")}
    return (hidden.cuda(), weight.cuda(), bias.cuda()), kwargs


@pytest.mark.parametrize("seed", cases.SEEDS)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:24] for c in op.CELLS])
def test_certified_cells(cell, seed, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every step within the rules in `_router_cases.RULES`; groups and ids exactly, in order."""
    certifying(op)
    args, kwargs = _build(cell.spec, seed)
    out = op(*args, **kwargs)
    ref = op.reference(*args, **kwargs)
    assert set(out) == set(ref)
    assert out["ids"].shape == (cell.spec["tokens"], cell.spec["top_k"])
    cases.assert_all_pass(cases.compare_routes(out, ref, "reference"))


@pytest.mark.parametrize(
    "case,match",
    [
        ("batched_3d", "hidden_states must be"),
        ("bias_length_one", "correction_bias must be"),
        ("fp64_bias", "correction_bias must be torch.float32"),
        ("top_k_beyond_kept_groups", "cannot be chosen"),
    ],
)
def test_guard_refuses_inputs_the_op_answers_wrongly(case, match) -> None:
    (hidden, weight, bias), kwargs = _build(op.CELLS[0].spec, seed=17)
    if case == "batched_3d":
        hidden = hidden.unsqueeze(0)
    elif case == "bias_length_one":
        bias = bias[:1]
    elif case == "fp64_bias":
        bias = bias.double()
    else:
        kwargs["top_k"] = kwargs["topk_group"] * 32 + 1
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(hidden, weight, bias, **kwargs)
