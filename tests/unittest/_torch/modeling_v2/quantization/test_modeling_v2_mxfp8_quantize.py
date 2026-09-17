# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_quantize catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (
    mxfp8_quantize as op,
)

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "mxfp8_quantize requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    x = _build(cell.spec, seed=abs(hash(cell.why)) % 2**31)
    swizzled, alignment = cell.spec["swizzled"], cell.spec["alignment"]
    with validating(op):
        data, scales = op(x, swizzled, alignment)
    ref_data, ref_scales = op.reference(x, swizzled, alignment)
    op.compare(data, ref_data)
    op.compare(scales, ref_scales)


def test_the_quantization_round_trips_within_the_block_bound() -> None:
    """Independent of the reference: dequantizing has to return the input.

    The cells compare two implementations of the same arithmetic, which would
    both be wrong together if the arithmetic itself were wrong. This bounds the
    error against the input instead: one power-of-two scale per 32 elements and
    e4m3's 3 mantissa bits give at worst 2^-3 relative on the block maximum.
    """
    x = torch.randn(64, 2880, dtype=torch.bfloat16, device="cuda")
    data, scales = op(x, False, 512)
    rows, blocks = 64, 3072 // 32
    scale = torch.exp2(scales.reshape(rows, blocks).float() - 127.0).unsqueeze(-1)
    deq = (data.float().view(rows, blocks, 32) * scale).view(rows, 3072)[:, :2880]
    block_amax = x.float().view(rows, 90, 32).abs().amax(-1, keepdim=True)
    err = (deq - x.float()).view(rows, 90, 32).abs().amax(-1, keepdim=True)
    assert torch.all(err <= block_amax * 2.0**-3 + 1e-6)


def test_guard_refuses_a_partial_block() -> None:
    """The kernel reads past the end of the last block rather than failing."""
    x = torch.randn(4, 2864, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="whole number of 32-element blocks"):
        with validating(op):
            op(x, False, 512)


def test_guard_refuses_an_alignment_that_is_not_a_block_multiple() -> None:
    """Pads to a width the scale count cannot describe."""
    x = torch.randn(4, 2880, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="multiple of the 32-element"):
        with validating(op):
            op(x, False, 48)


def test_guard_refuses_a_non_contiguous_input() -> None:
    """Read flat, so a strided view quantizes the wrong elements."""
    x = torch.randn(4, 5760, dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(AssertionError, match="contiguous"):
        with validating(op):
            op(x, False, 512)
