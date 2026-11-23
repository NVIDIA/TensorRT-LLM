# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm entry this target owns (`__r12`).

The entry owns the contract: `CELLS` is the certified range, `reference` is what
the kernel is compared against, and `is_valid` is the guard. This file owns what
a cell cannot carry -- turning a spec into real tensors, the comparison band and
its derivation, and the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash_for_conditional_generation.hf_003107f_orig__sm_103__unified.ops.norm.flashinfer_rmsnorm__r12 import (  # noqa: E501
    flashinfer_rmsnorm as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"

CELLS = op.CELLS

#: 3 ulp of the row's largest element and 0.5 ulp of relative RMS. Kernel and
#: reference compute the same fp32 math and round once to bf16, so a difference
#: is one bf16 rounding step of an element, read against the row's scale rather
#: than the element's own. Measured worst across the cells on sm_103: 1.36 /
#: 0.0053 ulp, and 1.06 / 0.0045 with the same cells re-seeded. The band is
#: tighter than torch's bf16 default (rtol 1.6e-2 is about 4.1 ulp), so no
#: wrong-variant control is needed to separate it from a wrong answer.
_ELEMENT_ULP = 3.0
_RMS_ULP = 0.5


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=g, device="cuda").to(dtype)
    weight = torch.randn(shape[-1], generator=g, device="cuda").to(dtype)
    return x, weight, spec["eps"]


def _cosine(out: torch.Tensor, ref: torch.Tensor) -> float:
    a, b = out.float().reshape(-1), ref.float().reshape(-1)
    return (a @ b / (a.norm() * b.norm())).item()


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, weight, eps = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    ref = op.reference(x, weight, eps)
    # Distribution first: a cell whose distribution matches but whose worst
    # element sits past a strict allclose is certified; a diverged one is not.
    cos = _cosine(out, ref)
    assert cos > 0.999999, f"cosine {cos:.9f}: the output diverged from the reference"
    assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """The hazard `is_valid` exists for: accepted, and silently plausible.

    Driving the op directly shows why the guard is not cosmetic -- it returns a
    tensor rather than raising, so a caller who forgot to dequantize gets a
    number back.
    """
    x = torch.randn(4, 3072, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(3072, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_rmsnorm(x, weight, 1e-6)
    assert unguarded.dtype is torch.float8_e4m3fn, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, weight, 1e-6)


def test_guard_refuses_a_strided_normalized_dim() -> None:
    """`x.stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 3072, 2, dtype=torch.bfloat16, device="cuda")
    x = buf[..., 0]
    assert x.stride(-1) != 1
    weight = torch.randn(3072, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, weight, 1e-6)
