# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate, `is_valid` is the guard. This file owns only what a
cell cannot carry -- turning a spec into real tensors -- plus the inputs the
guard exists to refuse.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm as op,
)

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(shape, generator=g, device="cuda").to(dtype)
    weight = torch.randn(shape[-1], generator=g, device="cuda").to(dtype)
    return x, weight, spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    x, weight, eps = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    with validating(op):
        out = op(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    op.compare(out, op.reference(x, weight, eps))


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """The hazard `is_valid` exists for: accepted, and silently plausible.

    Driving the op directly shows why the guard is not cosmetic -- it returns a
    tensor rather than raising, so a caller who forgot to dequantize gets a
    number back.
    """
    x = torch.randn(4, 2880, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(2880, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_rmsnorm(x, weight, 1e-6)
    assert unguarded.dtype is torch.float8_e4m3fn, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, weight, 1e-6)


def test_guard_refuses_a_strided_normalized_dim() -> None:
    """`x.stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 2880, 2, dtype=torch.bfloat16, device="cuda")
    x = buf[..., 0]
    assert x.stride(-1) != 1
    weight = torch.randn(2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, weight, 1e-6)
