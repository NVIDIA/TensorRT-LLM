# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the silu_and_mul operation of the glm5_next FP8 target.

The entry owns what is certified. This file turns a cell's spec into tensors, shows that the
cell data tell the clamp and the gate/up order apart, and drives the inputs the guard exists
to refuse.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.activation.silu_and_mul import (  # noqa: E501
    GLM_SWIGLU_LIMIT,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.activation.silu_and_mul import (  # noqa: E501
    silu_and_mul as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="silu_and_mul is certified on sm_100"
)

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """torch's bf16 band: rtol 1.6e-2 (about four bf16 ulp), atol 1e-5.

    Kernel and reference evaluate the same fp32 expression on identical bf16 inputs and round
    once. They differ only in the exponential inside the sigmoid (Triton's approximate exp
    against torch's), a few fp32 ulp, which can move a bf16 rounding by one ulp (0.4%
    relative) and no further. Where a very negative gate drives the product toward zero the
    two sit far inside the 1e-5 absolute floor. Each wrong reading this entry guards against
    -- no clamp, a symmetric gate clamp, the halves swapped -- misses by orders of magnitude on
    the elements it touches (`test_cell_data_tell_the_clamp_and_the_halves_apart`).
    """
    assert out.dtype == ref.dtype and out.shape == ref.shape, (out.dtype, ref.dtype)
    torch.testing.assert_close(out, ref)


def _build(spec: dict, seed: int) -> torch.Tensor:
    """FC1 outputs at the scale a GLM layer's look like: std 8, so the limit of 10 clamps
    about 11% of the gates (upper side) and 21% of the ups (both sides)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(spec["rows"], spec["cols"], generator=g, device=DEV) * 8.0
    return x.to(torch.bfloat16)


# The shapes the target calls this operation with, on top of the operation's own `CELLS`.
#
# GLM-5.3-Flash at TP4: the three dense layers' MLP (intermediate 12288 / 4 = 3072, so 2 x 3072
# wide) at every step, and the MoE layers' shared expert (2048 / 4 = 512) on steps the
# routed-expert decode path does not take (more than 4 rows). Generation steps are MTP-3
# verifies of 4 rows per request, up to 64 requests; a 13,312-row prefill runs both on every
# gathered row.
CELLS: tuple[Cell, ...] = op.CELLS + (
    Cell(
        why="dense MLP, MTP-3 verify of one request: 4 rows of 2 x 3072",
        spec=dict(rows=4, cols=6144, swiglu_limit=GLM_SWIGLU_LIMIT),
    ),
    Cell(
        why="dense MLP, MTP-3 verify of 64 requests: 256 rows of 2 x 3072",
        spec=dict(rows=256, cols=6144, swiglu_limit=GLM_SWIGLU_LIMIT),
    ),
    Cell(
        why="dense MLP over a 13,312-row prefill step",
        spec=dict(rows=13312, cols=6144, swiglu_limit=GLM_SWIGLU_LIMIT),
    ),
    Cell(
        why="shared expert, the smallest step past the 4-row decode path: 5 rows of 2 x 512",
        spec=dict(rows=5, cols=1024, swiglu_limit=GLM_SWIGLU_LIMIT),
    ),
    Cell(
        why="shared expert, MTP-3 verify of 64 requests: 256 rows of 2 x 512",
        spec=dict(rows=256, cols=1024, swiglu_limit=GLM_SWIGLU_LIMIT),
    ),
    Cell(
        why="shared expert over a 13,312-row prefill step",
        spec=dict(rows=13312, cols=1024, swiglu_limit=GLM_SWIGLU_LIMIT),
    ),
)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture
    certifying(op)
    x = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    limit = cell.spec["swiglu_limit"]
    out = op(x, swiglu_limit=limit)
    assert out.shape == (x.shape[0], x.shape[1] // 2) and out.dtype == x.dtype
    assert out.is_contiguous() and out.data_ptr() != x.data_ptr()
    _compare(out, op.reference(x, swiglu_limit=limit))


def test_cell_data_tell_the_clamp_and_the_halves_apart() -> None:
    """The cells certify the clamp and the half order only if their data can see them.

    At the cell scale each wrong reading misses: no clamp, the gate clamped on both sides
    (it is clamped on the upper side only), and the halves swapped ([up | gate] instead of
    [gate | up]).
    """
    spec = dict(rows=256, cols=1024, swiglu_limit=GLM_SWIGLU_LIMIT)
    x = _build(spec, seed=7)
    out = op(x, swiglu_limit=GLM_SWIGLU_LIMIT)
    _compare(out, op.reference(x, swiglu_limit=GLM_SWIGLU_LIMIT))

    d = x.shape[1] // 2
    gate, up = x[:, :d].float(), x[:, d:].float()
    lim = GLM_SWIGLU_LIMIT
    up_c = up.clamp(-lim, lim)
    symmetric = gate.clamp(-lim, lim)
    wrong = {
        "no clamp": op.reference(x),
        "symmetric gate clamp": (symmetric * torch.sigmoid(symmetric) * up_c).to(x.dtype),
        "halves swapped": op.reference(torch.cat([x[:, d:], x[:, :d]], dim=1), swiglu_limit=lim),
    }
    for name, ref in wrong.items():
        with pytest.raises(AssertionError):
            _compare(out, ref)
        print(f"  {name}: rejected")


def test_a_row_strided_view_is_read_correctly() -> None:
    """Rows are read through `x.stride(0)`, so a column slice of a wider buffer that keeps the
    last dim dense needs no copy -- the guard admits it."""
    buf = _build(dict(rows=8, cols=8192), seed=11)
    x = buf[:, :6144]
    assert x.stride(0) == 8192 and x.stride(1) == 1
    with validating(op):
        out = op(x, swiglu_limit=GLM_SWIGLU_LIMIT)
    _compare(out, op.reference(x.contiguous(), swiglu_limit=GLM_SWIGLU_LIMIT))


def test_guard_refuses_a_strided_last_dim() -> None:
    """The hazard: a strided last dim is read as dense and the answer is silently wrong."""
    buf = _build(dict(rows=8, cols=2048), seed=13)
    x = buf[:, ::2]
    assert x.stride(-1) == 2
    unguarded = op(x, swiglu_limit=GLM_SWIGLU_LIMIT)
    with pytest.raises(AssertionError):
        _compare(unguarded, op.reference(x, swiglu_limit=GLM_SWIGLU_LIMIT))
    with pytest.raises(AssertionError, match="last dim must be contiguous"):
        with validating(op):
            op(x, swiglu_limit=GLM_SWIGLU_LIMIT)


def test_guard_refuses_a_scale_without_a_dtype() -> None:
    x = _build(dict(rows=4, cols=1024), seed=17)
    scale = torch.ones(1, dtype=torch.float32, device=DEV)
    with pytest.raises(AssertionError, match="scale without dtype"):
        with validating(op):
            op(x, scale=scale, swiglu_limit=GLM_SWIGLU_LIMIT)
