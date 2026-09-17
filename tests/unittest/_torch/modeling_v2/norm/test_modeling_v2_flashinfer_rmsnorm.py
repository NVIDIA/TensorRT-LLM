# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry.

The entry owns what is certified: `flashinfer_rmsnorm.CELLS` is the list, and
`reference` / `compare` are the gate. This file owns only what a cell cannot
carry -- turning a spec into real tensors -- plus the hazards, which are
claims about inputs no cell describes.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"


def _build(spec, seed):
    """Materialize one cell's spec.

    `row_stride` asks for a last-dim-contiguous slice of a wider buffer, which
    is the only part of a spec that cannot be expressed as a plain `randn`.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape, dtype = spec["shape"], spec["dtype"]
    stride = spec.get("row_stride")
    if stride is None:
        x = torch.randn(shape, generator=g, device="cuda").to(dtype)
    else:
        assert len(shape) == 2 and stride > shape[1]
        buf = torch.randn((shape[0], stride), generator=g, device="cuda").to(dtype)
        x = buf[:, : shape[1]]
        assert not x.is_contiguous() and x.stride(-1) == 1
    weight = torch.randn(shape[-1], generator=g, device="cuda").to(dtype)
    return x, weight, spec["eps"]


@pytest.mark.parametrize(
    "cell", flashinfer_rmsnorm.CELLS, ids=[c.why[:40] for c in flashinfer_rmsnorm.CELLS]
)
def test_certified_cells(cell) -> None:
    """Every certified configuration, kernel against reference.

    Parametrized off the entry itself, so adding a cell there adds a case
    here and a cell that stops holding fails under its own `why`.
    """
    x, weight, eps = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    out = flashinfer_rmsnorm(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert out.data_ptr() != x.data_ptr(), "the op must return a new tensor"
    flashinfer_rmsnorm.compare(out, flashinfer_rmsnorm.reference(x, weight, eps))


def test_rejects_fp8_before_the_kernel_answers() -> None:
    """The hazard `is_valid` exists for: accepted, and silently plausible.

    Driving the op directly shows why the guard is not cosmetic -- it returns
    a tensor rather than raising, so a caller who forgot to dequantize gets a
    number back.
    """
    x = torch.randn(4, 4096, device="cuda").to(torch.float8_e4m3fn)
    weight = torch.randn(4096, device="cuda").to(torch.float8_e4m3fn)
    unguarded = torch.ops.trtllm.flashinfer_rmsnorm(x, weight, 1e-6)
    assert unguarded.dtype is torch.float8_e4m3fn, "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="dequantize"):
        flashinfer_rmsnorm(x, weight, 1e-6)


def test_rejects_a_strided_normalized_dim() -> None:
    """`x.stride(-1) != 1` reads the wrong elements rather than failing."""
    buf = torch.randn(8, 4096, 2, dtype=torch.bfloat16, device="cuda")
    x = buf[..., 0]
    assert x.stride(-1) != 1
    weight = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="contiguous"):
        flashinfer_rmsnorm(x, weight, 1e-6)
