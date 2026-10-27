# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_quantize catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.mxfp8_quantize import (  # noqa: E501
    mxfp8_quantize as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "mxfp8_quantize requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Bit-exact.

    Both sides compute the same power-of-two scale and the same e4m3
    rounding, so any difference is a defect rather than accumulated error.
    fp8 and uint8 both compare as bytes.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert torch.equal(out.view(torch.uint8), ref.view(torch.uint8)), (
        "quantization is deterministic; a mismatch is a defect, not drift"
    )


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(spec["shape"], generator=g, device="cuda").to(spec["dtype"])


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "gpt_oss/gpt_oss_120b__sm_10x__default": (
        Cell(
            why="gpt-oss hidden 2880, linear scales, aligned to 512 -> padded 3072",
            spec=dict(shape=(16, 2880), dtype=torch.bfloat16, swizzled=False, alignment=512),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    x = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    swizzled, alignment = cell.spec["swizzled"], cell.spec["alignment"]
    with validating(op):
        data, scales = op(x, swizzled, alignment)
    ref_data, ref_scales = op.reference(x, swizzled, alignment)
    _compare(data, ref_data)
    _compare(scales, ref_scales)


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


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="the Kimi K3 cell is certified on sm_100"
)
def test_kimi_k3_routed_hidden() -> None:
    torch.manual_seed(13)
    for t in (1, 8, 64, 65, 1024, 8192):
        x = torch.randn(t, 3584, dtype=torch.bfloat16, device="cuda")
        data, sf = op(x, False, 512)
        ref_data, ref_sf = op.reference(x, False, 512)
        assert data.shape == (t, 3584) and data.dtype == torch.float8_e4m3fn
        assert sf.shape == (t * 112,) and sf.dtype == torch.uint8
        assert torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8))
        assert torch.equal(sf, ref_sf)
        tight_data, tight_sf = op(x, False, 32)
        assert torch.equal(tight_data.view(torch.uint8), data.view(torch.uint8))
        assert torch.equal(tight_sf, sf)
