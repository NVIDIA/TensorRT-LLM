# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul entry this target owns (`__r18`).

The entry owns the contract: `CELLS` is the certified range, `reference` is what
the kernel is compared against, and `is_valid` is the guard. This file owns what
a cell cannot carry -- turning a spec into real tensors, the comparison band and
its derivation, and the inputs the guard exists to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.activation.flashinfer_silu_and_mul__r18 import (  # noqa: E501
    flashinfer_silu_and_mul as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"

CELLS = op.CELLS

#: 1 ulp of the row's largest element and 0.1 ulp of relative RMS. Kernel and
#: reference compute the same fp32 math and round once to bf16, so a difference
#: is one bf16 rounding step of an element, read against the row's scale rather
#: than the element's own. Measured worst across the cells on sm_103: bit-exact
#: (0 of 70,778,880 elements differ on the widest cell), 0.0 / 0.0 ulp. The band
#: is tighter than torch's bf16 default (rtol 1.6e-2 is about 4.1 ulp), so no
#: wrong-variant control is needed to separate it from a wrong answer.
_ELEMENT_ULP = 1.0
_RMS_ULP = 0.1


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


def _cosine(out: torch.Tensor, ref: torch.Tensor) -> float:
    a, b = out.float().reshape(-1), ref.float().reshape(-1)
    return (a @ b / (a.norm() * b.norm())).item()


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(x)
    assert out.shape == (*x.shape[:-1], x.shape[-1] // 2)
    assert out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    ref = op.reference(x)
    # Distribution first: a cell whose distribution matches but whose worst
    # element sits past a strict allclose is certified; a diverged one is not.
    cos = _cosine(out, ref)
    assert cos > 0.999999, f"cosine {cos:.9f}: the output diverged from the reference"
    assert_within_ulp(out, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


@pytest.mark.parametrize(
    "width,match",
    [
        (4095, "must be even"),
        # 24 bf16 elements is 48 bytes per row, which the op's own row check
        # accepts; the half is 24 bytes, so the second half starts mid-vector.
        (24, "16-byte vectors"),
        # The only width that reaches the third assert: for the half to be a
        # whole number of 16-byte vectors *and* smaller than one, it has to be
        # zero. Worth a case anyway -- it is the difference between a guard
        # that is unreachable and one that is merely narrow.
        (0, "at least one 16-byte vector"),
    ],
)
def test_guard_refuses_widths_the_shared_entry_refuses(width, match) -> None:
    """The guard is inherited from the shared entry and still fires here.

    Against the installed FlashInfer these widths would run: its launcher picks
    a vec_size dividing the half (`flashinfer/jit/activation.py`), so the guard
    is conservative rather than a hazard. It stays because this variant is the
    shared entry's contract at this target's shapes; no cell of this target
    reaches a width it refuses.
    """
    x = torch.randn(4, width, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match=match):
        with validating(op):
            op(x)
