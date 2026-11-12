# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp8_per_tensor_mm entry.

Cells on the qkv_proj K/V row slice of an FP8 weight (a row view of the whole
14336-row weight, as the target passes it) against the reference: saturating E4M3
quantize of `x / a`, fp32 product, `a * s`, one rounding. Beyond the cells: the
per-tensor scale guard.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.fp8_per_tensor_mm import (  # noqa: E501
    fp8_per_tensor_mm as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
QG = 12288  # Q + gate columns of qkv_proj; K/V follow


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """The quantized activation is the same (both multiply by the fp32 reciprocal of
    `a` and round to E4M3 with saturation); E4M3 x E4M3 products are exact in fp32;
    what is left is accumulation order and one rounding -- 4 units element-wise, 2 of
    RMS, as for every fp32-accumulating GEMM of this tree."""
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _build(m: int, k: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(14336, k, generator=g, device=DEV) * 0.02
    s = (w.abs().amax() / 448.0).float().reshape(())
    w8 = (w / s).to(torch.float8_e4m3fn)
    x = (torch.randn(m, k, generator=g, device=DEV) * 2.0).to(torch.bfloat16)
    # A static input scale that saturates the largest activations, as calibrated
    # scales do on outliers.
    a = (x.abs().amax().float() / 448.0 * 0.8).reshape(1)
    return x, w8[QG:], a, s.reshape(1)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    x, w8, a, ws = _build(s["m"], s["k"], seed_of(cell.why))
    assert w8.shape == (s["n"], s["k"]) and w8.is_contiguous()
    out = op(x, w8, a, ws)
    assert out.shape == (s["m"], s["n"])
    _compare(out, op.reference(x, w8, a, ws))


def test_guard_refuses_a_per_row_input_scale() -> None:
    x, w8, a, ws = _build(65, 5120, seed=1)
    with validating(op), pytest.raises(AssertionError, match="per-tensor"):
        op(x, w8, a.repeat(65), ws)
