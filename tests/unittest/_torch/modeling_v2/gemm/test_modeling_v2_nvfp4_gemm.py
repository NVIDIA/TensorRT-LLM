# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_gemm catalog entry."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.nvfp4_gemm import nvfp4_gemm as op
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)

__extra_import_path__ = [".."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "nvfp4_gemm requires a CUDA device"


def _operand(rows, k, seed):
    """Quantize a random operand the way a target does, via the fp4_quantize entry.

    Building operands with the sibling entry rather than by hand is deliberate:
    it is how a target produces them, so a disagreement between the two entries
    about the scale layout shows up here instead of in a model.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, generator=g, device="cuda").to(torch.bfloat16)
    gs = (448.0 * 6.0 / x.abs().max().float()).reshape(1)
    packed, sf = fp4_quantize(x, gs, 16, False, True)
    return packed, sf, gs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    seed = abs(hash(cell.why)) % 2**31
    act, act_sf, act_gs = _operand(spec["m"], spec["k"], seed)
    weight, weight_sf, weight_gs = _operand(spec["n"], spec["k"], seed + 1)
    # alpha folds both global scales back out of the product.
    alpha = (1.0 / (act_gs * weight_gs)).float().reshape(1)

    with validating(op):
        out = op(act, weight, act_sf, weight_sf, alpha, torch.bfloat16)
    assert out.shape == (spec["m"], spec["n"])
    op.compare(out, op.reference(act, weight, act_sf, weight_sf, alpha, torch.bfloat16))


def _ok():
    act, act_sf, act_gs = _operand(128, 512, 5)
    weight, weight_sf, weight_gs = _operand(256, 512, 6)
    alpha = (1.0 / (act_gs * weight_gs)).float().reshape(1)
    return act, weight, act_sf, weight_sf, alpha


def test_guard_refuses_a_strided_operand() -> None:
    """cublaslt ignores strides; cutlass raises. Which one answers is the op's choice."""
    act, weight, act_sf, weight_sf, alpha = _ok()
    strided = torch.zeros(128, 512, dtype=act.dtype, device="cuda")[:, ::2]
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(op):
            op(strided, weight, act_sf, weight_sf, alpha, torch.bfloat16)


def test_guard_refuses_a_multi_element_alpha() -> None:
    """Silently ignored past the first element; this build has no per-token alpha."""
    act, weight, act_sf, weight_sf, _ = _ok()
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(op):
            op(act, weight, act_sf, weight_sf, torch.ones(128, device="cuda"), torch.bfloat16)


def test_guard_refuses_a_short_scale_buffer() -> None:
    """Read past its end: the kernel indexes the padded rectangle, not the real one."""
    act, weight, act_sf, weight_sf, alpha = _ok()
    with pytest.raises(AssertionError, match="act_sf must hold at least"):
        with validating(op):
            op(act, weight, act_sf[:-4], weight_sf, alpha, torch.bfloat16)
