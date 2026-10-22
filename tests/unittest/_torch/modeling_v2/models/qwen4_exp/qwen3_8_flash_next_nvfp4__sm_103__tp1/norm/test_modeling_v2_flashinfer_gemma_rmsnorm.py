# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_gemma_rmsnorm entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.norm.flashinfer_gemma_rmsnorm import (  # noqa: E501
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
    elif "row_step" in spec:
        buffer = randn(shape[0] * spec["row_step"], width)
        x = buffer[:: spec["row_step"]]
    else:
        buffer = x = randn(*shape)
    weight = (torch.randn(width, generator=g, device="cuda") * spec.get("weight_scale", 1.0)).to(
        dtype
    )
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


def test_weight_bias_is_one_not_zero() -> None:
    """Discrimination: the scale is (1 + w), not w.

    Measured: 0 away from the (1 + w) reference, ~3.98 from the plain-w one.
    """
    _, x, weight, eps = _build(dict(shape=(4, 5120), dtype=torch.bfloat16, eps=1e-6), seed=0)
    out = op(x, weight, eps)
    xf = x.float()
    plain = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)
    gemma = op.reference(x, weight, eps)

    torch.testing.assert_close(out, gemma)
    gemma_dev = (out.float() - gemma.float()).abs().max().item()
    plain_dev = (out.float() - plain.float()).abs().max().item()
    assert plain_dev > 100 * max(gemma_dev, 2.0**-8), (gemma_dev, plain_dev)


def test_guard_refuses_fp8_before_the_kernel_answers() -> None:
    """Both fp8 dtypes are in the DSL dtype table: accepted, and a tensor comes back."""
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        x = torch.randn(4, 2560, device="cuda").to(dtype)
        w = torch.randn(2560, device="cuda").to(dtype)
        unguarded = torch.ops.trtllm.flashinfer_gemma_rmsnorm(x, w, 1e-6)
        assert unguarded.dtype is dtype and unguarded.shape == x.shape, (
            "the hazard this guard covers is gone"
        )
        with pytest.raises(AssertionError, match="dequantize"):
            with validating(op):
                op(x, w, 1e-6)


def test_unsupported_dtypes_raise() -> None:
    for dtype in (torch.float64, torch.int8, torch.uint8):
        x = torch.ones(4, 2560, dtype=dtype, device="cuda")
        w = torch.ones(2560, dtype=dtype, device="cuda")
        with pytest.raises(KeyError):
            op(x, w, 1e-6)


def test_shape_dtype_and_device_mismatches_raise() -> None:
    """All loud, which is why `is_valid` restates none of them."""
    w = torch.randn(2560, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(4, 2560, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        op(x, w.float(), 1e-6)
    with pytest.raises(ValueError):
        op(x, torch.randn(2048, dtype=torch.bfloat16, device="cuda"), 1e-6)
    with pytest.raises(ValueError):
        op(x, w.cpu(), 1e-6)
    with pytest.raises(ValueError):
        op(x.cpu(), w.cpu(), 1e-6)
    with pytest.raises(ValueError):
        op(torch.randn(2560, dtype=torch.bfloat16, device="cuda"), w, 1e-6)
    # a strided normalized dim: "Mismatched mX.strides[1] ... expected to be 1"
    with pytest.raises(ValueError, match="strides"):
        op(torch.randn(8, 2560, 2, dtype=torch.bfloat16, device="cuda")[..., 0], w, 1e-6)
    with pytest.raises(ValueError):
        op(
            torch.randn(2, 4, 8, 128, dtype=torch.bfloat16, device="cuda"),
            torch.randn(128, dtype=torch.bfloat16, device="cuda"),
            1e-6,
        )
