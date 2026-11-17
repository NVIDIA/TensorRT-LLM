# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_silu_and_mul entry of the MiniMax-M3 tp8 target: SwiGLU-OAI and plain SwiGLU."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.activation.m3_silu_and_mul import (  # noqa: E501
    m3_silu_and_mul as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_silu_and_mul requires a CUDA device"

OAI = (7.0, 1.702, 1.0)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Gate and up spread past the clamp (|x| up to ~12) so both clamps engage.
    x = (torch.randn(spec["m"], 2 * spec["inter"], generator=g, device="cuda") * 4.0).to(
        torch.bfloat16
    )
    return (x, *OAI) if spec["oai"] else (x, None, None, None)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    """fp32 arithmetic and one bf16 rounding on both sides: the kernel's fast sigmoid moves the fp32 value by a few
    fp32 ulps, which flips the bf16 rounding of an element by at most one step (1 unit of ULP[bf16] of the row);
    measured on GB300: worst element 0.5 unit, relative RMS 0.01."""
    args = _build(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(*args)
    assert out.shape == (cell.spec["m"], cell.spec["inter"]) and out.dtype == torch.bfloat16
    assert_within_ulp(out, op.reference(*args), element_ulp=1.0, rms_ulp=0.1)


def test_the_gate_sees_a_missing_clamp() -> None:
    """Dropping the SwiGLU-OAI clamps moves every element whose gate or up passed the limit far outside the band."""
    spec = op.CELLS[0].spec
    x, *oai = _build(dict(spec, m=64), 3)
    out = op(x, *oai)
    with pytest.raises(AssertionError):
        assert_within_ulp(out, op.reference(x, None, oai[1], oai[2]), element_ulp=1.0, rms_ulp=0.1)
