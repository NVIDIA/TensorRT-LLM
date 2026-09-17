# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the noaux_tc_op catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op as op

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "noaux_tc_op requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        spec["tokens"], spec["experts"], generator=g, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
    bias = torch.randn(spec["experts"], generator=g, device="cuda", dtype=torch.float32)
    return logits, bias, spec["n_group"], spec["topk_group"], spec["topk"], spec["scale"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    with validating(op):
        weights, ids = op(*args)
    ref_weights, ref_ids = op.reference(*args)

    # Ids are an exact claim: routing either picks the same experts or it does
    # not, and a near-miss here is a different set of experts, not a rounding.
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert ids.dtype is torch.int32
    op.compare(weights, ref_weights)


def test_the_sigmoid_is_the_tanh_form() -> None:
    """The entry's reference claims a specific sigmoid; hold the op to it.

    `1/(1+exp(-x))` and `0.5*tanh(x/2)+0.5` agree everywhere a random logit
    lands and diverge in the tails, where the tanh form saturates exactly. A
    cell over random inputs cannot tell them apart, so this drives the tails
    directly -- otherwise `reference` could encode the wrong sigmoid and every
    cell would still pass.
    """
    experts = 8
    logits = torch.full((4, experts), -30.0, dtype=torch.bfloat16, device="cuda")
    logits[:, 0] = 30.0
    bias = torch.zeros(experts, dtype=torch.float32, device="cuda")
    weights, _ = op(logits, bias, 1, 1, 2, 1.0)
    ref_weights, _ = op.reference(logits, bias, 1, 1, 2, 1.0)
    op.compare(weights, ref_weights)


def _ok():
    logits = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(16, dtype=torch.float32, device="cuda")
    return logits, bias


def test_guard_refuses_strided_logits() -> None:
    """Read as a dense buffer from data_ptr(); strides are ignored."""
    logits = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")[:, ::2]
    _, bias = _ok()
    with pytest.raises(AssertionError, match="router_logits must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_a_strided_bias() -> None:
    logits, _ = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")[::2]
    with pytest.raises(AssertionError, match="bias must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_topk_past_the_expert_count() -> None:
    """Reads out of bounds and emits ids outside the expert range."""
    logits, bias = _ok()
    with pytest.raises(AssertionError, match="exceeds num_experts"):
        with validating(op):
            op(logits, bias, 1, 1, 17, 1.0)
