# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_gemma_rmsnorm entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.norm.flashinfer_gemma_rmsnorm import (  # noqa: E501
    flashinfer_gemma_rmsnorm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(x, weight, eps)` for one spec: a contiguous `x`."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    x = torch.randn(*shape, generator=g, device="cuda").to(dtype)
    weight = torch.randn(shape[-1], generator=g, device="cuda").to(dtype)
    return x, weight, spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, weight, eps = _build(cell.spec, seed=_seed(cell.why))
    assert tuple(x.shape) == cell.spec["shape"] and x.is_contiguous()
    x_before = x.clone()

    out = op(x, weight, eps)

    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    assert torch.equal(x, x_before), "x is read-only"
    torch.testing.assert_close(out, op.reference(x, weight, eps))


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """fp8 is in the DSL dtype table: accepted, and a tensor comes back."""
    x = torch.randn(4, 5120, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(5120, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_gemma_rmsnorm(x, w, 1e-6)
    assert unguarded.dtype is x.dtype and unguarded.shape == x.shape, (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, w, 1e-6)
