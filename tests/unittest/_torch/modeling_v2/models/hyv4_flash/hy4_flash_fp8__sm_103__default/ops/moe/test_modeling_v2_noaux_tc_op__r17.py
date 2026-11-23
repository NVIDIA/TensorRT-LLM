# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Hy4-Flash target's noaux_tc_op variant (`__r17`).

The entry is a thin wrapper over a compiled-in C++ thop, so what is certified
here is the thop's routing at this checkpoint's shape: 512 experts in one
group, top-16, scaling 3.998, bf16 logits and a bf16 `e_score_correction_bias`.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.moe.noaux_tc_op__r17 import (  # noqa: E501
    noaux_tc_op as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "noaux_tc_op requires a CUDA device"

CELLS = op.CELLS

# The kernel and the reference compute the same fp32 numerator -- the selected
# unbiased sigmoid scores, bit-identical -- over a denominator that differs
# only in accumulation order: an fp32 warp butterfly in the kernel against
# torch's fp32 row sum, then the same fp64 division. That moves the quotient
# by ~1e-7 relative, so an element can differ only where it straddles a bf16
# rounding midpoint -- and then by exactly one bf16 ulp. One ulp of a value in
# a row is between 2^-8 and 2^-7 of the row's largest magnitude, while
# `assert_within_ulp`'s unit is 2^-8 of that magnitude, so one rounding step
# is at most 2.0 of its units. Measured on sm_103 over these cells: 0 differing
# elements at 1, 37 and 64 tokens; 1 of 81920 at 5120 tokens, one true ulp
# (1.44 units) with a relative RMS of 1.3e-5 (0.003 units). 0.05 rms units is
# 15x that and still 0.3x torch's bf16 default rtol (1.6e-2).
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.05


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        spec["tokens"], spec["experts"], generator=g, device="cuda", dtype=torch.float32
    ).to(spec["logits_dtype"])
    bias = torch.randn(spec["experts"], generator=g, device="cuda", dtype=torch.float32).to(
        spec["bias_dtype"]
    )
    return logits, bias, spec["n_group"], spec["topk_group"], spec["topk"], spec["scale"]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 - pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    args = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    weights, ids = op(*args)
    ref_weights, ref_ids = op.reference(*args)

    # Ids are an exact claim: routing either picks the same experts or it does
    # not, and a near-miss here is a different set of experts, not a rounding.
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert ids.dtype is torch.int32
    assert weights.dtype is spec["logits_dtype"], "the weights follow the logits' dtype"

    # Structural claims the reference comparison cannot make on its own.
    assert ids.shape == (spec["tokens"], spec["topk"])
    assert weights.shape == ids.shape
    assert weights.device == ids.device == args[0].device
    assert weights.is_contiguous() and ids.is_contiguous()
    assert int(ids.min()) >= 0 and int(ids.max()) < spec["experts"], "an id names no expert"
    assert (torch.sort(ids, dim=-1).values.diff(dim=-1) > 0).all(), "an expert is named twice"
    assert bool((weights > 0).all()), "a selected expert has zero combine weight"
    # Every row's weights sum to the scaling factor: the renormalization. The
    # bf16 rounding of 16 weights near 3.998/16 moves the sum by at most
    # 16 * half an ulp of 0.25, ~8e-3 absolute or 2e-3 relative; measured worst
    # 2.1e-3. 1e-2 leaves 5x.
    row_sum = weights.float().sum(-1)
    torch.testing.assert_close(
        row_sum, torch.full_like(row_sum, spec["scale"]), rtol=1e-2, atol=0.0
    )
    assert_within_ulp(weights, ref_weights, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_the_sigmoid_is_the_tanh_form() -> None:
    """The entry's reference claims a specific sigmoid; hold the op to it.

    `1/(1+exp(-x))` and `0.5*tanh(x/2)+0.5` agree everywhere a random logit
    lands and diverge in the tails, where the tanh form saturates exactly. A
    cell over random inputs cannot tell them apart, so this drives the tails
    directly. Exact equality, not a band: at x = -30 the logistic form leaves
    9.4e-14 in the second weight -- representable in bf16, and admitted by the
    dtype default's 1e-5 atol.
    """
    experts = 8
    logits = torch.full((4, experts), -30.0, dtype=torch.bfloat16, device="cuda")
    logits[:, 0] = 30.0
    bias = torch.zeros(experts, dtype=torch.bfloat16, device="cuda")
    weights, ids = op(logits, bias, 1, 1, 2, 1.0)
    ref_weights, ref_ids = op.reference(logits, bias, 1, 1, 2, 1.0)
    assert torch.equal(ids, ref_ids)
    assert torch.equal(weights, ref_weights)
    assert float(weights[:, 1].max()) == 0.0, "the tail weight is not exactly zero"


def test_the_bias_steers_but_does_not_weigh() -> None:
    """A bias large enough to force an expert in must not reach its weight.

    The 512-expert router with this target's dtypes. The misreading that
    gathers weights from `sigmoid + bias` keeps the same experts and changes
    every weight, which is what `test_the_band_rejects_biased_weights` holds
    the band to.
    """
    g = torch.Generator(device="cuda").manual_seed(17)
    logits = torch.randn(16, 512, generator=g, device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    bias = torch.zeros(512, dtype=torch.bfloat16, device="cuda")
    bias[17] = 100.0
    weights, ids = op(logits, bias, 1, 1, 16, 3.998)
    assert bool((ids == 17).any(dim=1).all()), "the biased expert was not selected"
    ref_weights, ref_ids = op.reference(logits, bias, 1, 1, 16, 3.998)
    assert torch.equal(ids, ref_ids)
    assert_within_ulp(weights, ref_weights, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_the_band_rejects_biased_weights() -> None:
    """The band must reject the misreading the entry warns about.

    Same selected experts, weights gathered from `sigmoid + bias` instead of
    `sigmoid`: a wrong variant a loose band would admit.
    """
    spec = dict(op.CELLS[1].spec)
    args = _build(spec, seed=11)
    logits, bias, _, _, topk, scale = args
    weights, ids = op(*args)
    scores = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    biased = scores + bias.float()
    biased_ids = torch.sort(biased, dim=-1, descending=True, stable=True).indices[:, :topk]
    assert torch.equal(biased_ids.to(torch.int32), ids), "the control must keep the same experts"
    gathered = torch.gather(biased, 1, biased_ids)
    gathered = (gathered.double() / gathered.sum(-1, keepdim=True).double() * scale).to(
        logits.dtype
    )
    with pytest.raises(AssertionError):
        assert_within_ulp(weights, gathered, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_guard_refuses_strided_logits() -> None:
    """The kernel reads data_ptr() as a dense buffer; strides are ignored.

    Demonstrated first: the same allocation read as a 4x16 dense matrix does
    not answer what its strided view means.
    """
    g = torch.Generator(device="cuda").manual_seed(5)
    base = torch.randn(4, 32, generator=g, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(16, generator=g, device="cuda", dtype=torch.bfloat16)
    weights, ids = op(base[:, ::2], bias, 1, 1, 8, 1.0)
    ref_weights, ref_ids = op.reference(base[:, ::2].contiguous(), bias, 1, 1, 8, 1.0)
    assert not (torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)), (
        "the unguarded strided call was expected to answer from the dense buffer"
    )
    with pytest.raises(AssertionError, match="router_logits must be contiguous"):
        with validating(op):
            op(base[:, ::2], bias, 1, 1, 8, 1.0)


def test_guard_refuses_a_strided_bias() -> None:
    g = torch.Generator(device="cuda").manual_seed(6)
    logits = torch.randn(4, 16, generator=g, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(32, generator=g, device="cuda", dtype=torch.bfloat16)
    weights, ids = op(logits, base[::2], 1, 1, 8, 1.0)
    ref_weights, ref_ids = op.reference(logits, base[::2].contiguous(), 1, 1, 8, 1.0)
    assert not (torch.equal(ids, ref_ids) and torch.equal(weights, ref_weights)), (
        "the unguarded strided bias was expected to be read as a dense buffer"
    )
    with pytest.raises(AssertionError, match="bias must be contiguous"):
        with validating(op):
            op(logits, base[::2], 1, 1, 8, 1.0)


def test_guard_refuses_topk_past_the_expert_count() -> None:
    """Demonstrated first: the kernel answers 17 experts from a 16-expert row.

    A contiguous buffer with slack after it, so the out-of-bounds read stays
    inside an allocation; the 17th id is 16, an expert that does not exist.
    """
    g = torch.Generator(device="cuda").manual_seed(7)
    pad = torch.randn(4 * 16 + 64, generator=g, device="cuda", dtype=torch.bfloat16)
    logits = pad[: 4 * 16].view(4, 16)
    bias = torch.randn(16, generator=g, device="cuda", dtype=torch.bfloat16)
    _, ids = op(logits, bias, 1, 1, 17, 1.0)
    assert ids.shape == (4, 17), "the unguarded call answered 17 experts from 16"
    assert int(ids.max()) >= 16, "the unguarded call emitted an id no expert has"
    with pytest.raises(AssertionError, match="exceeds num_experts"):
        with validating(op):
            op(logits, bias, 1, 1, 17, 1.0)
