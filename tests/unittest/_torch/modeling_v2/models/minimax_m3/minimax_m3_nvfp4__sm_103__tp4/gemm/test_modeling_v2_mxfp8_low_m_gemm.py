# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mxfp8_low_m_gemm entry of the MiniMax-M3 tp4 target."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.gemm.mxfp8_low_m_gemm import (  # noqa: E501
    mxfp8_low_m_gemm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "mxfp8_low_m_gemm requires a CUDA device"


def _swizzle(byte: torch.Tensor) -> torch.Tensor:
    """[N, K / 32] scale bytes into the 128x4-swizzled buffer of ``mxfp8_quantize(w, True)``."""
    rows, cols = byte.shape
    padded_cols, padded_rows = (cols + 3) // 4 * 4, (rows + 127) // 128 * 128
    r = torch.arange(rows, device=byte.device).view(-1, 1)
    c = torch.arange(cols, device=byte.device).view(1, -1)
    index = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * (128 * padded_cols)
    )
    out = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8, device=byte.device)
    out[index.reshape(-1)] = byte.reshape(-1)
    return out


def _build(spec, seed):
    """A BF16 activation and an MXFP8 weight: E4M3 values of a unit normal, UE8M0 block scales 2^-9 .. 2^-3."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, n, k = spec["m"], spec["n"], spec["k"]
    x = torch.randn(m, k, generator=g, device="cuda").to(torch.bfloat16)
    w = torch.randn(n, k, generator=g, device="cuda").to(torch.float8_e4m3fn)
    sf = torch.randint(118, 125, (n, k // 32), generator=g, device="cuda", dtype=torch.int32)
    return x, w, _swizzle(sf.to(torch.uint8))


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Accumulation order only, then one BF16 rounding.

    Both sides multiply the same E4M3 operands and power-of-two scales; the kernel sums each 32-wide block in the
    MMA, rescales it exactly and accumulates blocks per warp, warps in order and split-K ranks in order, the
    reference sums in a single fp32 GEMM. The fp32 results differ by a few fp32 ulps of the row's magnitude, which
    moves the BF16 rounding by at most one step of the element -- at most 2 units of ``ULP[bf16]`` against the row
    scale. Measured on GB300 over every cell: worst element 0.32 unit, relative RMS 0.006 unit.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.25)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    x, w, wsf = _build(spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    args = (x, w, wsf, spec["warps"], spec["cluster_k"], spec["row_tiles"], False)
    with validating(op):
        out = op(*args)
    assert out.shape == (spec["m"], spec["n"]) and out.dtype == torch.bfloat16
    _compare(out, op.reference(*args))


def test_the_gate_sees_a_wrong_scale_layout() -> None:
    """Block scales read in the wrong order land far outside the band: most outputs pick up other blocks'
    scales, which a layout bug in either the op or the reference would do."""
    spec = op.CELLS[1].spec
    g = torch.Generator(device="cuda").manual_seed(7)
    x, w, wsf = _build(spec, seed=7)
    scrambled = wsf[torch.randperm(wsf.numel(), generator=g, device="cuda")]
    args = (x, w, wsf, spec["warps"], spec["cluster_k"], spec["row_tiles"], False)
    out = op(*args)
    with pytest.raises(AssertionError):
        _compare(out, op.reference(x, w, scrambled, *args[3:]))


def test_guard_refuses_a_short_scale_buffer() -> None:
    """The kernel bulk-copies scale chunks by computed offset; a linear-length buffer is read past its end."""
    spec = op.CELLS[0].spec
    x, w, wsf = _build(spec, seed=11)
    n, k = w.shape
    with pytest.raises(AssertionError, match="swizzled UE8M0 buffer"):
        with validating(op):
            op(
                x,
                w,
                wsf[: n * k // 32 - 512],
                spec["warps"],
                spec["cluster_k"],
                spec["row_tiles"],
                False,
            )
