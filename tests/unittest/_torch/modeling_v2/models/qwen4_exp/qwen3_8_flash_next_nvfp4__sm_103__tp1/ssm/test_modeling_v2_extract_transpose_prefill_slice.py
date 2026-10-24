# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the extract_transpose_prefill_slice entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ssm.extract_transpose_prefill_slice import (  # noqa: E501
    extract_transpose_prefill_slice as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "extract_transpose_prefill_slice requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bitwise: the kernel moves elements with no cast and no arithmetic,
    so any tolerance would only hide a wrong index. Compared as raw bytes so
    dtypes without an equality kernel (`float8_e4m3fn`) are gated the same."""
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    if out.numel():
        assert torch.equal(out.reshape(-1).view(torch.uint8), ref.reshape(-1).view(torch.uint8))


def _src(num_rows: int, num_cols: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(num_rows, num_cols, generator=g, device="cuda").to(dtype)


def _build(spec, seed):
    src = _src(spec["num_rows"], spec["num_cols"], spec["dtype"], seed)
    if "row_stop" in spec:
        src = src[: spec["row_stop"]]
    if "col_slice" in spec:
        lo, hi = spec["col_slice"]
        src = src[:, lo:hi]
        assert not src.is_contiguous() and src.stride(1) == 1
    return src, spec["n"], spec["start_col"], spec["width"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    src, n, start_col, width = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = src.contiguous().view(torch.uint8).clone()
    out = op(src, n, start_col, width)
    assert out.data_ptr() != src.data_ptr() or out.numel() == 0
    assert torch.equal(src.contiguous().view(torch.uint8), before), "src is read-only"
    _compare(out, op.reference(src, n, start_col, width))


def test_guard_refuses_a_range_outside_src() -> None:
    """The kernel masks only against the requested extents and reads past
    `src` rather than failing; the guard bounds both axes (probed here at 128
    rows of a 64-row `src`)."""
    src = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="outside src"):
        with validating(op):
            op(src, 128, 0, 256)
