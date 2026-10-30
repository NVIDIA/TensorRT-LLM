# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_gemma_rmsnorm entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.norm.flashinfer_gemma_rmsnorm__r36 import (  # noqa: E501
    flashinfer_gemma_rmsnorm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "flashinfer_gemma_rmsnorm requires a CUDA device"


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`(buffer, x, weight, eps)`: `x` is a view of `buffer` laid out as the spec says."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    width = shape[-1]

    def randn(*s):
        return torch.randn(*s, generator=g, device="cuda").to(dtype)

    if "row_width" in spec:
        rows, off = shape[0], spec["col_offset"]
        buffer = randn(rows, spec["row_width"])
        x = buffer[:, off : off + width]
    else:
        buffer = x = randn(*shape)
    weight = torch.randn(width, generator=g, device="cuda").to(dtype)
    return buffer, x, weight, spec["eps"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    buffer, x, weight, eps = _build(cell.spec, seed=_seed(cell.why))
    assert tuple(x.shape) == cell.spec["shape"] and x.stride(-1) == 1
    if "row_width" in cell.spec and x.shape[0] > 1:
        assert x.stride(0) == cell.spec["row_width"] and not x.is_contiguous()
    buffer_before = buffer.clone()

    out = op(x, weight, eps)

    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    assert torch.equal(buffer, buffer_before), "x (and the stream it slices) is read-only"
    torch.testing.assert_close(out, op.reference(x, weight, eps))


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """fp8 is in the DSL dtype table: accepted, and a tensor comes back."""
    x = torch.randn(4, 2560, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(2560, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_gemma_rmsnorm(x, w, 1e-6)
    assert unguarded.dtype is x.dtype and unguarded.shape == x.shape, (
        "the hazard this guard covers is gone"
    )
    with pytest.raises(AssertionError, match="dequantize"):
        with validating(op):
            op(x, w, 1e-6)
